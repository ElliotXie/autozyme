/* knn.c — approximate kNN graph for the low-dim single-cell embedding regime.
 * RP-tree forest init + NN-Descent refinement, race-free parallel (double-buffer
 * + new/old pruning) with a NEON fixed-d L2. Beats pynndescent/hnswlib on the
 * PCA-embedding regime (see include/scblas/knn.h). Fork-safe: pthreads, no BLAS.
 */
#include <scblas/knn.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static long scblas_online_cpus(void) {
#ifdef _WIN32
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (si.dwNumberOfProcessors > 0) ? (long)si.dwNumberOfProcessors : 1L;
#else
  long h = sysconf(_SC_NPROCESSORS_ONLN);
  return (h > 0) ? h : 1L;
#endif
}

static inline uint64_t sm64(uint64_t *s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
  z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }

static inline float sqd(const float *a, const float *b, int d){
#if defined(__ARM_NEON)
  float32x4_t a0=vdupq_n_f32(0),a1=a0; int t=0;   /* 2 accumulators: hide FMA latency, light reduction */
  for(;t+8<=d;t+=8){
    float32x4_t e0=vsubq_f32(vld1q_f32(a+t),  vld1q_f32(b+t));   a0=vfmaq_f32(a0,e0,e0);
    float32x4_t e1=vsubq_f32(vld1q_f32(a+t+4),vld1q_f32(b+t+4)); a1=vfmaq_f32(a1,e1,e1);
  }
  for(;t+4<=d;t+=4){ float32x4_t e=vsubq_f32(vld1q_f32(a+t),vld1q_f32(b+t)); a0=vfmaq_f32(a0,e,e); }
  float s=vaddvq_f32(vaddq_f32(a0,a1)); for(;t<d;++t){ float e=a[t]-b[t]; s+=e*e; } return s;
#else
  float s=0; for(int t=0;t<d;++t){ float e=a[t]-b[t]; s+=e*e; } return s;
#endif
}
static inline int hpush(float *hd, int32_t *hi, uint8_t *hf, int k, int32_t c, float dc){
  if(dc>=hd[0]) return 0; for(int j=0;j<k;++j) if(hi[j]==c) return 0;
  hd[0]=dc; hi[0]=c; hf[0]=1; int i=0;
  for(;;){ int l=2*i+1,r=2*i+2,m=i; if(l<k&&hd[l]>hd[m])m=l; if(r<k&&hd[r]>hd[m])m=r; if(m==i)break;
    float td=hd[i];hd[i]=hd[m];hd[m]=td; int32_t ti=hi[i];hi[i]=hi[m];hi[m]=ti;
    uint8_t tf=hf[i];hf[i]=hf[m];hf[m]=tf; i=m; } return 1;
}

typedef struct{ int n,d,k,leaf; const float*X; float*hd; int32_t*hi; uint8_t*hf; double*pv; } rpx;
/* NEON fixed-d dot product (for the RP-tree split projection) */
static inline float dotp(const float *a, const float *b, int d){
#if defined(__ARM_NEON)
  float32x4_t a0=vdupq_n_f32(0.0f),a1=a0; int t=0;
  for(;t+8<=d;t+=8){
    a0=vfmaq_f32(a0,vld1q_f32(a+t),  vld1q_f32(b+t));
    a1=vfmaq_f32(a1,vld1q_f32(a+t+4),vld1q_f32(b+t+4));
  }
  for(;t+4<=d;t+=4) a0=vfmaq_f32(a0, vld1q_f32(a+t), vld1q_f32(b+t));
  float s=vaddvq_f32(vaddq_f32(a0,a1)); for(;t<d;++t) s+=a[t]*b[t]; return s;
#else
  float s=0; for(int t=0;t<d;++t) s+=a[t]*b[t]; return s;
#endif
}
/* quickselect: partition pv[0..n) (idx[] in lockstep) so pv[target] is in sorted position and
   everything before it is <= — O(n) average median split, no qsort, no comparator callback. */
static void qselect(double *pv, int32_t *idx, int n, int target, uint64_t *rng){
  int lo=0, hi=n-1;
  while(lo<hi){
    int p=lo+(int)(sm64(rng)%(uint64_t)(hi-lo+1)); double pivot=pv[p];
    { double td=pv[p];pv[p]=pv[lo];pv[lo]=td; int32_t ti=idx[p];idx[p]=idx[lo];idx[lo]=ti; }
    int store=lo;
    for(int i=lo+1;i<=hi;++i) if(pv[i]<pivot){ ++store;
      double td=pv[store];pv[store]=pv[i];pv[i]=td; int32_t ti=idx[store];idx[store]=idx[i];idx[i]=ti; }
    { double td=pv[lo];pv[lo]=pv[store];pv[store]=td; int32_t ti=idx[lo];idx[lo]=idx[store];idx[store]=ti; }
    if(store==target) return; else if(store<target) lo=store+1; else hi=store-1;
  }
}
static void rp(rpx*c, int32_t*idx, int lo, int hi, uint64_t*rng){
  int n=hi-lo, d=c->d;
  if(n<=c->leaf){ for(int a=lo;a<hi;++a) for(int b=a+1;b<hi;++b){ int u=idx[a],v=idx[b];
      float dd=sqd(c->X+(size_t)u*d,c->X+(size_t)v*d,d);
      hpush(c->hd+(size_t)u*c->k,c->hi+(size_t)u*c->k,c->hf+(size_t)u*c->k,c->k,v,dd);
      hpush(c->hd+(size_t)v*c->k,c->hi+(size_t)v*c->k,c->hf+(size_t)v*c->k,c->k,u,dd); } return; }
  int ia=(int)(sm64(rng)%(uint64_t)n), ib; do{ ib=(int)(sm64(rng)%(uint64_t)n);}while(ib==ia);
  const float*pa=c->X+(size_t)idx[lo+ia]*d, *pb=c->X+(size_t)idx[lo+ib]*d;
  float dir[d];                                    /* split direction pa->pb (low-d by design) */
  for(int t=0;t<d;++t) dir[t]=pb[t]-pa[t];
  double*pv=c->pv;                                  /* per-thread scratch (no per-node malloc); offset dropped (cancels in median) */
  for(int i=0;i<n;++i) pv[i]=(double)dotp(c->X+(size_t)idx[lo+i]*d, dir, d);
  qselect(pv, idx+lo, n, n/2, rng);                 /* O(n) median partition, idx in lockstep — no qsort */
  int mid=lo+n/2;
  rp(c,idx,lo,mid,rng); rp(c,idx,mid,hi,rng);
}

/* parallel RP-tree forest (thread-local heaps) */
typedef struct{ int t0,t1,n,d,k,leaf; const float*X; float*hd; int32_t*hi; uint8_t*hf; uint64_t seed; } treework;
static void *treeworker(void*arg){
  treework*w=(treework*)arg; int n=w->n,k=w->k;
  for(size_t i=0;i<(size_t)n*k;++i){ w->hd[i]=1e30f; w->hi[i]=-1; w->hf[i]=1; }
  int32_t*idx=malloc((size_t)n*sizeof(int32_t)); double*pvbuf=malloc((size_t)n*sizeof(double)); uint64_t rng=w->seed;
  rpx c={n,w->d,k,w->leaf,w->X,w->hd,w->hi,w->hf,pvbuf};
  for(int t=w->t0;t<w->t1;++t){ for(int i=0;i<n;++i)idx[i]=i;
    for(int i=n-1;i>0;--i){ int j=(int)(sm64(&rng)%(uint64_t)(i+1)); int tt=idx[i];idx[i]=idx[j];idx[j]=tt; }
    rp(&c,idx,0,n,&rng); }
  free(idx); free(pvbuf); return NULL;
}
/* parallel merge of per-thread forest heaps into the main heap */
typedef struct{ int i0,i1,k,P; float**thd; int32_t**thi; float*hd; int32_t*hi; uint8_t*hf; } mergework;
static void *mergeworker(void*arg){
  mergework*w=(mergework*)arg; int k=w->k,P=w->P;
  for(int i=w->i0;i<w->i1;++i){ float*hdi=w->hd+(size_t)i*k; int32_t*hii=w->hi+(size_t)i*k; uint8_t*hfi=w->hf+(size_t)i*k;
    for(int j=0;j<k;++j){ hdi[j]=1e30f; hii[j]=-1; hfi[j]=1; }
    for(int e=0;e<P;++e){ const float*sd=w->thd[e]+(size_t)i*k; const int32_t*si=w->thi[e]+(size_t)i*k;
      for(int j=0;j<k;++j) if(si[j]>=0) hpush(hdi,hii,hfi,k,si[j],sd[j]); } }
  return NULL;
}
/* parallel NN-descent (double-buffer self-update, new/old pruning) */
typedef struct{ int i0,i1,n,d,k; const float*X; const int32_t*sh; const uint8_t*sf;
  const int32_t*revn; const int*rcn; float*hd; int32_t*hi; uint8_t*hf; long upd; } work;
static void *descworker(void*arg){
  work*w=(work*)arg; int k=w->k,d=w->d,n=w->n; const int32_t*SH=w->sh; const uint8_t*SF=w->sf; long upd=0;
  for(int i=w->i0;i<w->i1;++i){
    const float*xi=w->X+(size_t)i*d; float*hdi=w->hd+(size_t)i*k; int32_t*hii=w->hi+(size_t)i*k; uint8_t*hfi=w->hf+(size_t)i*k;
    for(int a=0;a<k;++a){ if(!SF[(size_t)i*k+a])continue; int32_t j=SH[(size_t)i*k+a]; if(j<0||j>=n)continue;
      const int32_t*Sj=SH+(size_t)j*k;
      for(int b=0;b<k;++b){ int32_t c=Sj[b]; if(c<0||c>=n||c==i)continue;
        float dd=sqd(xi,w->X+(size_t)c*d,d); upd+=hpush(hdi,hii,hfi,k,c,dd); } }
    for(int a=0;a<w->rcn[i];++a){ int32_t c=w->revn[(size_t)i*k+a]; if(c<0||c>=n||c==i)continue;
      float dd=sqd(xi,w->X+(size_t)c*d,d); upd+=hpush(hdi,hii,hfi,k,c,dd); } }
  w->upd=upd; return NULL;
}

SCBLAS_API int scblas_knn_descent_f32(int n_cells, int n_dims, int k, const float*X,
    int n_trees, int n_iters, int leaf_size, uint64_t seed,
    int32_t*knn_idx, float*knn_dist, int n_threads){
  int n=n_cells, d=n_dims;
  if(n<=0||d<=0||k<=0||k>=n) return -1;
  if(n_trees<1)n_trees=1; if(n_iters<1)n_iters=1; if(leaf_size<k+1)leaf_size=k+1;
  uint8_t*hf=malloc((size_t)n*k);
  int32_t*sh=malloc((size_t)n*k*sizeof(int32_t)); uint8_t*sf=malloc((size_t)n*k);
  int32_t*revn=malloc((size_t)n*k*sizeof(int32_t)); int*rcn=malloc((size_t)n*sizeof(int));
  if(!hf||!sh||!sf||!revn||!rcn){ free(hf);free(sh);free(sf);free(revn);free(rcn); return -2; }
  float*hd=knn_dist; int32_t*hi=knn_idx;   /* build directly into the caller's output */

  int eff=n_threads; if(eff<=0){ long h=scblas_online_cpus(); eff=(h>0)?(int)h:1; }
  int teff=eff>n_trees?n_trees:eff; if(teff<1)teff=1;
  /* ---- parallel RP-tree forest + parallel merge ---- */
  {
    pthread_t*th=malloc((size_t)teff*sizeof(pthread_t)); treework*tw=malloc((size_t)teff*sizeof(treework));
    float**thd=malloc((size_t)teff*sizeof(float*)); int32_t**thi=malloc((size_t)teff*sizeof(int32_t*)); uint8_t**thf=malloc((size_t)teff*sizeof(uint8_t*));
    if(!th||!tw||!thd||!thi||!thf){ free(th);free(tw);free(thd);free(thi);free(thf);free(hf);free(sh);free(sf);free(revn);free(rcn); return -2; }
    int base=n_trees/teff, rem=n_trees%teff, st=0, sp=0;
    for(int e=0;e<teff;++e){ int cnt=base+(e<rem?1:0);
      thd[e]=malloc((size_t)n*k*sizeof(float)); thi[e]=malloc((size_t)n*k*sizeof(int32_t)); thf[e]=malloc((size_t)n*k);
      tw[e]=(treework){st,st+cnt,n,d,k,leaf_size,X,thd[e],thi[e],thf[e],seed^(0x9E37ULL*(uint64_t)(e+1))}; st+=cnt;
      if(cnt==0)continue; if(pthread_create(&th[sp],NULL,treeworker,&tw[e])==0)sp++; else treeworker(&tw[e]); }
    for(int e=0;e<sp;++e)pthread_join(th[e],NULL);
    int meff=eff>n?n:eff; if(meff<1)meff=1;
    pthread_t*mt=malloc((size_t)meff*sizeof(pthread_t)); mergework*mw=malloc((size_t)meff*sizeof(mergework));
    int mb=n/meff, mr=n%meff, ms=0, msp=0;
    for(int e=0;e<meff;++e){ int cnt=mb+(e<mr?1:0); mw[e]=(mergework){ms,ms+cnt,k,teff,thd,thi,hd,hi,hf}; ms+=cnt;
      if(cnt==0)continue; if(pthread_create(&mt[msp],NULL,mergeworker,&mw[e])==0)msp++; else mergeworker(&mw[e]); }
    for(int e=0;e<msp;++e)pthread_join(mt[e],NULL);
    for(int e=0;e<teff;++e){ free(thd[e]);free(thi[e]);free(thf[e]); }
    free(thd);free(thi);free(thf);free(th);free(tw);free(mt);free(mw);
  }
  /* ---- parallel NN-descent ---- */
  int deff=eff>n?n:eff; if(deff<1)deff=1;
  pthread_t*th=malloc((size_t)deff*sizeof(pthread_t)); work*wk=malloc((size_t)deff*sizeof(work));
  if(th&&wk) for(int it=0;it<n_iters;++it){
    memcpy(sh,hi,(size_t)n*k*sizeof(int32_t)); memcpy(sf,hf,(size_t)n*k); memset(hf,0,(size_t)n*k);
    memset(rcn,0,(size_t)n*sizeof(int));
    for(int i=0;i<n;++i) for(int j=0;j<k;++j){ if(!sf[(size_t)i*k+j])continue; int32_t v=sh[(size_t)i*k+j];
      if(v>=0&&v<n&&rcn[v]<k) revn[(size_t)v*k+rcn[v]++]=i; }
    int base=n/deff, rem=n%deff, start=0, sp=0;
    for(int e=0;e<deff;++e){ int cnt=base+(e<rem?1:0);
      wk[e]=(work){start,start+cnt,n,d,k,X,sh,sf,revn,rcn,hd,hi,hf,0}; start+=cnt;
      if(cnt==0)continue; if(pthread_create(&th[sp],NULL,descworker,&wk[e])==0)sp++; else descworker(&wk[e]); }
    long upd=0; for(int e=0;e<sp;++e)pthread_join(th[e],NULL); for(int e=0;e<deff;++e)upd+=wk[e].upd;
    if(upd<(long)((double)n*k*0.0002)) break;
  }
  free(th);free(wk); free(hf);free(sh);free(sf);free(revn);free(rcn);
  return 0;
}
