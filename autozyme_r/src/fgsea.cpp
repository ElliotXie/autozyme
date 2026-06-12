// fgsea.cpp — vendored fgseaMultilevelCpp matching INSTALLED fgsea 1.34.2
// (Bioconductor RELEASE_3_21 branch). Double-based internal representation.
//
// Provides two kernels used by inst/patches/fgsea.R:
//   1. calcEsLeBatchCpp                — batch ES + leading-edge over a list of
//      pre-prepped (sorted desc, abs^p) selected gene index sets. Replaces the
//      `lapply(pathwaysFiltered, calcGseaStat, ..., returnLeadingEdge=TRUE)`
//      step in fgseaMultilevel's initial scoring (~5s for 9420 paths x 16 cl).
//   2. fastFgseaMultilevelBatchCpp     — std::thread-parallel batch over
//      multilevel-size-groups. Each group is independent (own EsRuler, own
//      RNG seeded with the shared seed → bit-exact vs per-group entry).
//
// Bit-exact RNG path: std::mt19937 seeded with the same int seed as upstream
// → same uid_wrapper / std::shuffle sequences → identical currentSamples
// / NS evolution → identical pval and isCpGeHalf output.

// [[Rcpp::depends(BH)]]

#include <Rcpp.h>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <utility>
#include <string>

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

using std::vector;
using std::pair;
using std::make_pair;
using std::min;
using std::max;
using std::sort;
using std::lower_bound;
using std::upper_bound;
using std::nth_element;
using std::fill;
using std::swap;
using std::log;
using std::exp;
using std::sqrt;
using std::abs;
using std::shuffle;
using std::mt19937;

// ---------- calcEsLeBatchCpp ----------
// Batch ES + leading-edge kernel. Replaces the per-pathway
// `lapply(pathwaysFiltered, calcGseaStat, ...)` step in fgseaMultilevel.
// Math identical to upstream's calcGseaStat (cumulative statistic
// = cumsum(abs(stats[S])^p)/NR - (S-rank)/(N-k)), with the scoreType branch
// identical too. Caller passes already-prepped (sorted desc, abs^p) stats
// from preparePathwaysAndStats; gseaParam is therefore not re-applied here.
//
// Returns List(es = NumericVector[m], le = List of IntegerVector[1-based idx
// into stats], same shape as upstream gseaStatRes[, 'leadingEdge']).
// [[Rcpp::export]]
Rcpp::List calcEsLeBatchCpp(Rcpp::NumericVector const & stats,
                            Rcpp::List const & selectedGenes,
                            std::string scoreType) {
  int N = stats.size();
  int m = selectedGenes.size();
  Rcpp::NumericVector es(m);
  Rcpp::List le(m);
  for (int i = 0; i < m; ++i) {
    Rcpp::IntegerVector S0 = selectedGenes[i];
    int k = S0.size();
    std::vector<int> S(k);
    for (int j = 0; j < k; ++j) S[j] = S0[j];
    std::sort(S.begin(), S.end());

    double NR = 0.0;
    for (int j = 0; j < k; ++j) NR += stats[S[j]-1];

    double cumSum = 0.0;
    double maxTop = -1e308, minBot = 1e308;
    int    iMaxTop = 0, iMinBot = 0;
    double Nm = (double)(N - k);

    for (int j = 0; j < k; ++j) {
      double cur = stats[S[j]-1];
      double base = (double)(S[j] - (j+1)) / Nm;
      double bottom, top;
      if (NR == 0.0) {
        cumSum += 1.0 / k;
        bottom  = (cumSum - 1.0/k) - base;
        top     = cumSum - base;
      } else {
        cumSum += cur / NR;
        bottom  = (cumSum - cur/NR) - base;
        top     = cumSum - base;
      }
      if (top > maxTop)    { maxTop = top; iMaxTop = j+1; }
      if (bottom < minBot) { minBot = bottom; iMinBot = j+1; }
    }

    bool useTop = false;
    bool empty  = false;
    if (scoreType == "std") {
      if (maxTop > -minBot)      { es[i] = maxTop; useTop = true; }
      else if (maxTop < -minBot) { es[i] = minBot; useTop = false; }
      else                       { es[i] = 0.0; empty = true; }
    } else if (scoreType == "pos") {
      es[i] = maxTop; useTop = true;
    } else {
      es[i] = minBot; useTop = false;
    }

    if (empty) {
      le[i] = Rcpp::IntegerVector::create();
    } else if (useTop) {
      Rcpp::IntegerVector leIdx(iMaxTop);
      for (int j = 0; j < iMaxTop; ++j) leIdx[j] = S[j];
      le[i] = leIdx;
    } else {
      int sz = k - iMinBot + 1;
      Rcpp::IntegerVector leIdx(sz);
      for (int j = 0; j < sz; ++j) leIdx[j] = S[k - 1 - j];
      le[i] = leIdx;
    }
  }
  return Rcpp::List::create(Rcpp::Named("es") = es, Rcpp::Named("le") = le);
}

// ---------- esCalculation (verbatim from fgsea 1.34.x src/esCalculation.cpp) ----------
static double calcES(const vector<double> &ranks, const vector<int> &p, double NS) {
  int n = (int) ranks.size();
  int k = (int) p.size();
  double res = 0.0;
  double cur = 0.0;
  double q1 = 1.0 / (n - k);
  double q2 = 1.0 / NS;
  int last = -1;
  for (int pos : p) {
    cur -= q1 * (pos - last - 1);
    if (abs(cur) > abs(res)) res = cur;
    cur += q2 * ranks[pos];
    if (abs(cur) > abs(res)) res = cur;
    last = pos;
  }
  return res;
}
static double calcES(const vector<double> &ranks, const vector<int> &p) {
  double NS = 0.0;
  for (int pos : p) NS += ranks[pos];
  return calcES(ranks, p, NS);
}
static double calcPositiveES(const vector<double> &ranks, const vector<int> &p, double NS) {
  int n = (int) ranks.size();
  int k = (int) p.size();
  double res = 0.0;
  double cur = 0.0;
  double q1 = 1.0 / (n - k);
  double q2 = 1.0 / NS;
  int last = -1;
  for (int pos : p) {
    cur += q2 * ranks[pos] - q1 * (pos - last - 1);
    res = max(res, cur);
    last = pos;
  }
  return res;
}
static double calcPositiveES(const vector<double> &ranks, const vector<int> &p) {
  double NS = 0.0;
  for (int pos : p) NS += ranks[pos];
  return calcPositiveES(ranks, p, NS);
}

// ---------- util (verbatim from fgsea 1.34.x src/util.cpp) ----------
struct uid_wrapper {
  int from, len;
  mt19937& rng;
  unsigned completePart;
  uid_wrapper(int _from, int _to, mt19937& _rng)
    : from(_from), len(_to - _from + 1), rng(_rng) {
    unsigned maxVal = rng.max();
    completePart = maxVal - maxVal % len;
  }
  int operator()() {
    unsigned x;
    do { x = rng(); } while (x >= completePart);
    return from + x % len;
  }
};

static vector<int> combination(const int &a, const int &b, const int &k, mt19937& rng) {
  uid_wrapper uni(a, b, rng);
  vector<int> v;
  v.reserve(k);
  int n = b - a + 1;
  vector<char> used(n);
  if (k < n * 1.0 / 2) {
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < 100; j++) {
        int x = uni();
        if (!used[x - a]) {
          v.push_back(x);
          used[x - a] = true;
          break;
        }
      }
    }
  } else {
    for (int r = n - k; r < n; ++r) {
      int x = uid_wrapper(0, r, rng)();
      if (!used[x]) { v.push_back(a + x); used[x] = true; }
      else          { v.push_back(a + r); used[r] = true; }
    }
    shuffle(v.begin(), v.end(), rng);
  }
  return v;
}

static double betaMeanLog(unsigned long a, unsigned long b) {
  return boost::math::digamma(a) - boost::math::digamma(b + 1);
}

// ---------- EsRuler (verbatim from fgsea 1.34.x src/fgseaMultilevelSupplement.{h,cpp}) ----------
static pair<double, bool> calcLogCorrection(const vector<unsigned int> &probCorrector,
                                            long probCorrIndx, unsigned int sampleSize) {
  double result = 0.0;
  unsigned long halfSize = (sampleSize + 1) / 2;
  unsigned long remainder = sampleSize - probCorrIndx % (halfSize);
  double condProb = betaMeanLog(probCorrector[probCorrIndx] + 1, remainder);
  result += condProb;
  if (exp(condProb) >= 0.5) return make_pair(result, true);
  return make_pair(result, false);
}

class EsRuler {
private:
  const vector<double> &ranks;
  const unsigned int sampleSize;
  const unsigned int pathwaySize;

  vector<double> enrichmentScores;
  vector<vector<int>> currentSamples;
  vector<unsigned int> probCorrector;

  void duplicateSamples();

  vector<int> chunkLastElement;
  int chunksNumber;

  struct SampleChunks {
    vector<double> chunkSum;
    vector<vector<int>> chunks;
    SampleChunks(int n) : chunkSum(n), chunks(n) {}
  };

  int perturbate(const vector<double> &ranks, int k, SampleChunks &sampleChunks,
                 double bound, mt19937 &rng);
  int chunkLen(int ind) {
    return (ind == 0) ? chunkLastElement[0] : chunkLastElement[ind] - chunkLastElement[ind - 1];
  }

public:
  EsRuler(const vector<double> &inpRanks, unsigned int inpSampleSize, unsigned int inpPathwaySize)
    : ranks(inpRanks), sampleSize(inpSampleSize), pathwaySize(inpPathwaySize) {
    currentSamples.resize(inpSampleSize);
  }
  ~EsRuler() = default;
  void extend(double ES, int seed, double eps);
  pair<double, bool> getPvalue(double ES, double eps, bool sign);
};

void EsRuler::duplicateSamples() {
  vector<pair<double, int>> stats(sampleSize);
  vector<int> posEsIndxs;
  int totalPosEsCount = 0;
  for (int sampleId = 0; sampleId < (int)sampleSize; sampleId++) {
    double sampleEsPos = calcPositiveES(ranks, currentSamples[sampleId]);
    double sampleEs    = calcES(ranks, currentSamples[sampleId]);
    if (sampleEs > 0) { totalPosEsCount++; posEsIndxs.push_back(sampleId); }
    stats[sampleId] = make_pair(sampleEsPos, sampleId);
  }
  sort(stats.begin(), stats.end());
  for (int sampleId = 0; 2 * sampleId < (int)sampleSize; sampleId++) {
    enrichmentScores.push_back(stats[sampleId].first);
    if (std::find(posEsIndxs.begin(), posEsIndxs.end(), stats[sampleId].second) != posEsIndxs.end()) {
      totalPosEsCount--;
    }
    probCorrector.push_back(totalPosEsCount);
  }
  vector<vector<int>> new_sets;
  for (int sampleId = 0; 2 * sampleId < (int)sampleSize - 2; sampleId++) {
    for (int rep = 0; rep < 2; rep++) {
      new_sets.push_back(currentSamples[stats[sampleSize - 1 - sampleId].second]);
    }
  }
  new_sets.push_back(currentSamples[stats[sampleSize >> 1].second]);
  swap(currentSamples, new_sets);
}

void EsRuler::extend(double ES, int seed, double eps) {
  mt19937 gen(seed);
  for (int sampleId = 0; sampleId < (int)sampleSize; sampleId++) {
    currentSamples[sampleId] = combination(0, ranks.size() - 1, pathwaySize, gen);
    sort(currentSamples[sampleId].begin(), currentSamples[sampleId].end());
    (void) calcES(ranks, currentSamples[sampleId]);  // upstream computes but ignores
  }
  chunksNumber = max(1, (int) sqrt((double) pathwaySize));
  chunkLastElement = vector<int>(chunksNumber);
  chunkLastElement[chunksNumber - 1] = ranks.size();
  vector<int> tmp(sampleSize);
  vector<SampleChunks> samplesChunks(sampleSize, SampleChunks(chunksNumber));

  duplicateSamples();
  while (enrichmentScores.back() <= ES - 1e-10) {
    for (int i = 0, pos = 0; i < chunksNumber - 1; ++i) {
      pos += (pathwaySize + i) / chunksNumber;
      for (int j = 0; j < (int)sampleSize; ++j) tmp[j] = currentSamples[j][pos];
      nth_element(tmp.begin(), tmp.begin() + sampleSize / 2, tmp.end());
      chunkLastElement[i] = tmp[sampleSize / 2];
    }
    for (int i = 0; i < (int)sampleSize; ++i) {
      fill(samplesChunks[i].chunkSum.begin(), samplesChunks[i].chunkSum.end(), 0.0);
      for (int j = 0; j < chunksNumber; ++j) samplesChunks[i].chunks[j].clear();
      int cnt = 0;
      for (int pos : currentSamples[i]) {
        while (chunkLastElement[cnt] <= pos) ++cnt;
        samplesChunks[i].chunks[cnt].push_back(pos);
        samplesChunks[i].chunkSum[cnt] += ranks[pos];
      }
    }
    for (int moves = 0; moves < (int)sampleSize * (int)pathwaySize; ) {
      for (int sampleId = 0; sampleId < (int)sampleSize; sampleId++) {
        moves += perturbate(ranks, pathwaySize, samplesChunks[sampleId],
                            enrichmentScores.back(), gen);
      }
    }
    for (int i = 0; i < (int)sampleSize; ++i) {
      currentSamples[i].clear();
      for (int j = 0; j < chunksNumber; ++j)
        for (int pos : samplesChunks[i].chunks[j]) currentSamples[i].push_back(pos);
    }
    double prevTopScore = enrichmentScores.back();
    duplicateSamples();
    if (enrichmentScores.back() <= prevTopScore) break;
    if (eps != 0) {
      unsigned long k = enrichmentScores.size() / ((sampleSize + 1) / 2);
      if (k > -log2(0.5 * eps)) break;
    }
  }
}

pair<double, bool> EsRuler::getPvalue(double ES, double eps, bool sign) {
  unsigned long halfSize = (sampleSize + 1) / 2;
  auto it = enrichmentScores.begin();
  bool goodError = true;
  if (ES >= enrichmentScores.back()) {
    it = enrichmentScores.end() - 1;
    if (ES > enrichmentScores.back() + 1e-10) goodError = false;
  } else {
    it = lower_bound(enrichmentScores.begin(), enrichmentScores.end(), ES);
  }
  unsigned long indx = 0;
  (it - enrichmentScores.begin()) > 0 ? (indx = (it - enrichmentScores.begin())) : indx = 0;
  unsigned long k = indx / halfSize;
  unsigned long remainder = sampleSize - (indx % halfSize);
  double adjLog = betaMeanLog(halfSize, sampleSize);
  double adjLogPval = k * adjLog + betaMeanLog(remainder + 1, sampleSize);
  if (sign) {
    return make_pair(max(0.0, min(1.0, exp(adjLogPval))), goodError);
  }
  pair<double, bool> correction = calcLogCorrection(probCorrector, indx, sampleSize);
  double resLog = adjLogPval + correction.first;
  return make_pair(max(0.0, min(1.0, exp(resLog))), goodError && correction.second);
}

int EsRuler::perturbate(const vector<double> &ranks, int k, SampleChunks &sampleChunks,
                        double bound, mt19937 &rng) {
  double pertPrmtr = 0.1;
  int n = (int) ranks.size();
  uid_wrapper uid_n(0, n - 1, rng);
  uid_wrapper uid_k(0, k - 1, rng);
  double NS = 0;
  for (double cs : sampleChunks.chunkSum) NS += cs;
  double q1 = 1.0 / (n - k);
  int iters = max(1, (int)(k * pertPrmtr));
  int moves = 0;

  int candVal = -1;
  bool hasCand = false;
  int candX = 0;
  double candY = 0;

  for (int i = 0; i < iters; i++) {
    int oldInd = uid_k();
    int oldChunkInd = 0, oldIndInChunk = 0;
    int oldVal;
    {
      int tmp = oldInd;
      while ((int) sampleChunks.chunks[oldChunkInd].size() <= tmp) {
        tmp -= sampleChunks.chunks[oldChunkInd].size();
        ++oldChunkInd;
      }
      oldIndInChunk = tmp;
      oldVal = sampleChunks.chunks[oldChunkInd][oldIndInChunk];
    }
    int newVal = uid_n();
    int newChunkInd = 0;
    {
      int sz = (int) chunkLastElement.size();
      while (newChunkInd < sz && chunkLastElement[newChunkInd] <= newVal) ++newChunkInd;
    }
    int newIndInChunk = 0;
    {
      auto& nc = sampleChunks.chunks[newChunkInd];
      int sz = (int) nc.size();
      while (newIndInChunk < sz && nc[newIndInChunk] < newVal) ++newIndInChunk;
    }
    if (newIndInChunk < (int) sampleChunks.chunks[newChunkInd].size() && sampleChunks.chunks[newChunkInd][newIndInChunk] == newVal) {
      if (newVal == oldVal) ++moves;
      continue;
    }
    sampleChunks.chunks[oldChunkInd].erase(sampleChunks.chunks[oldChunkInd].begin() + oldIndInChunk);
    sampleChunks.chunks[newChunkInd].insert(
      sampleChunks.chunks[newChunkInd].begin() + newIndInChunk - (oldChunkInd == newChunkInd && oldIndInChunk < newIndInChunk ? 1 : 0),
      newVal);
    NS = NS - ranks[oldVal] + ranks[newVal];
    sampleChunks.chunkSum[oldChunkInd] -= ranks[oldVal];
    sampleChunks.chunkSum[newChunkInd] += ranks[newVal];

    if (hasCand) { if (oldVal == candVal) hasCand = false; }
    if (hasCand) {
      if (oldVal < candVal) { candX++; candY -= ranks[oldVal]; }
      if (newVal < candVal) { candX--; candY += ranks[newVal]; }
    }
    double q2 = 1.0 / NS;
    if (hasCand && -q1 * candX + q2 * candY > bound) { ++moves; continue; }

    int curX = 0;
    double curY = 0;
    bool ok = false;
    int last = -1;
    for (int i2 = 0; i2 < (int) sampleChunks.chunks.size(); ++i2) {
      if (q2 * (curY + sampleChunks.chunkSum[i2]) - q1 * curX < bound) {
        curY += sampleChunks.chunkSum[i2];
        curX += chunkLastElement[i2] - last - 1 - (int) sampleChunks.chunks[i2].size();
        last = chunkLastElement[i2] - 1;
      } else {
        for (int pos : sampleChunks.chunks[i2]) {
          curY += ranks[pos];
          curX += pos - last - 1;
          if (q2 * curY - q1 * curX > bound) {
            ok = true;
            hasCand = true;
            candX = curX; candY = curY; candVal = pos;
            break;
          }
          last = pos;
        }
        if (ok) break;
        curX += chunkLastElement[i2] - 1 - last;
        last = chunkLastElement[i2] - 1;
      }
    }
    if (!ok) {
      NS = NS - ranks[newVal] + ranks[oldVal];
      sampleChunks.chunkSum[oldChunkInd] += ranks[oldVal];
      sampleChunks.chunkSum[newChunkInd] -= ranks[newVal];
      sampleChunks.chunks[newChunkInd].erase(
        sampleChunks.chunks[newChunkInd].begin() + newIndInChunk - (oldChunkInd == newChunkInd && oldIndInChunk < newIndInChunk ? 1 : 0));
      sampleChunks.chunks[oldChunkInd].insert(sampleChunks.chunks[oldChunkInd].begin() + oldIndInChunk, oldVal);
      if (hasCand) { if (newVal == candVal) hasCand = false; }
      if (hasCand) {
        if (oldVal < candVal) { candX--; candY += ranks[oldVal]; }
        if (newVal < candVal) { candX++; candY -= ranks[newVal]; }
      }
    } else {
      ++moves;
    }
  }
  return moves;
}

// ---------- Batched entry: take all multilevel-size-groups for a cluster ----------
// Each group is independent (own EsRuler, own RNG seeded with shared seed).
// std::thread parallel — Apple clang ships libomp separately, so use C++11
// threads to keep the build dependency-free.
//
// [[Rcpp::export]]
Rcpp::List fastFgseaMultilevelBatchCpp(Rcpp::List groupES,
                                       Rcpp::IntegerVector pathwaySizes,
                                       Rcpp::NumericVector eps_per_group,
                                       const Rcpp::NumericVector& ranks,
                                       int sampleSize, int seed, bool sign,
                                       int nthreads = 0) {
  int nGroups = groupES.size();

  // Marshal once (serial — Rcpp not thread-safe).
  std::vector<std::vector<double>> all_es(nGroups);
  for (int i = 0; i < nGroups; ++i) all_es[i] = Rcpp::as<std::vector<double>>(groupES[i]);
  std::vector<int>    sizes(pathwaySizes.begin(), pathwaySizes.end());
  std::vector<double> epsv(eps_per_group.begin(), eps_per_group.end());
  std::vector<double> posRanks = Rcpp::as<std::vector<double>>(ranks);
  for (int i = 0; i < (int)posRanks.size(); ++i) posRanks[i] = abs(posRanks[i]);
  std::vector<double> negRanks(posRanks.rbegin(), posRanks.rend());

  // LPT-style ordering: dispatch largest pathwaySize groups first.
  std::vector<int> dispatch_order(nGroups);
  for (int i = 0; i < nGroups; ++i) dispatch_order[i] = i;
  std::sort(dispatch_order.begin(), dispatch_order.end(),
            [&](int a, int b) { return sizes[a] > sizes[b]; });

  std::vector<std::vector<double>> outP(nGroups);
  std::vector<std::vector<int>>    outICH(nGroups);

  int nthr = nthreads;
  if (nthr <= 0) nthr = 1;
  if (nthr > nGroups) nthr = nGroups;
  std::atomic<int> next{0};
  auto worker = [&]() {
    for (;;) {
      int idx = next.fetch_add(1, std::memory_order_relaxed);
      if (idx >= nGroups) break;
      int i = dispatch_order[idx];
      const std::vector<double>& es = all_es[i];
      int psize = sizes[i];
      double epsLocal = epsv[i];

      EsRuler rulerPos(posRanks, sampleSize, psize);
      EsRuler rulerNeg(negRanks, sampleSize, psize);
      double maxES = *std::max_element(es.begin(), es.end());
      double minES = *std::min_element(es.begin(), es.end());
      if (maxES >= 0) rulerPos.extend(abs(maxES), seed, epsLocal);
      if (minES < 0)  rulerNeg.extend(abs(minES), seed, epsLocal);

      int nrow = (int) es.size();
      std::vector<double> pv(nrow);
      std::vector<int>    ich(nrow);
      for (int j = 0; j < nrow; ++j) {
        double cur = es[j];
        pair<double, bool> rp = (cur >= 0)
          ? rulerPos.getPvalue(abs(cur), epsLocal, sign)
          : rulerNeg.getPvalue(abs(cur), epsLocal, sign);
        pv[j]  = rp.first;
        ich[j] = rp.second ? 1 : 0;
      }
      outP[i]   = std::move(pv);
      outICH[i] = std::move(ich);
    }
  };
  if (nthr == 1) {
    worker();
  } else {
    std::vector<std::thread> threads;
    threads.reserve(nthr - 1);
    for (int t = 1; t < nthr; ++t) threads.emplace_back(worker);
    worker();
    for (auto& th : threads) th.join();
  }

  // Marshal back (serial).
  Rcpp::List results(nGroups);
  for (int i = 0; i < nGroups; ++i) {
    Rcpp::LogicalVector ich(outICH[i].size());
    for (int j = 0; j < (int)outICH[i].size(); ++j) ich[j] = (outICH[i][j] != 0);
    results[i] = Rcpp::DataFrame::create(
      Rcpp::Named("cppMPval")      = outP[i],
      Rcpp::Named("cppIsCpGeHalf") = ich);
  }
  return results;
}
