import numpy as np

class Individual(list):
    def __init__(self,genome):
        for i in genome:
            self.append(i)
        self.score=-np.inf
    def get_score(self):
        return self.score

class Population(list):
    def __init__(self, n_pop, n_genes, pool, 
                 score_func, cross_func, mut_func,
                 death_ratio=0.5, mut_prob=0.05, collective_eval=False,
                 random_state=None):
        self._np = n_pop
        self._ng = n_genes
        self._dr = death_ratio
        self._mp = mut_prob
        self._pool = pool
        self._seed = random_state
        self._scoref = score_func
        self._crossf = cross_func
        self._mutf = mut_func
        self._ceval=collective_eval
        if random_state:
            self._rng=np.random.default_rng(seed=random_state)
        else:
            self._rng=np.random.default_rng()
        for n in range(self._np):
            self.append(Individual(self._rng.choice(pool,size=self._ng)))
        # make initial eval
        if self._ceval:
            scores = score_func(self)
            for i,s in enumerate(scores):
                self[i].score=s
        else:
            for i in self:
                i.score = score_func(i)
        self.sort()
    def _iterate(self):
        # number of survivals
        nsurv = int(len(self)*(1.0-self._dr))
        # get mating odds
        #self.sort()
        scores = np.array([i.score for i in self[:nsurv]])
        scores = np.sqrt( (scores - scores.max())**2)
        scores = np.cumsum(scores)
        scores = scores / scores.max()
        scores = np.insert(scores,0,0)
        #print(scores)
        #print(nsurv ,np.arange(nsurv, len(self)))
        if scores[1]>0.999:
            print("Population stagnated.")
            raise ValueError("Population Stagnated")
        for n in np.arange(nsurv, len(self)):
            x = self._rng.uniform()
            p1 = np.argwhere(scores<=x)[-1][0]-1
            p2 = p1
            while(p2==p1):
                x = self._rng.uniform()
                p2 = np.argwhere(scores<=x)[-1][0]-1
            self[n]=self._crossf(self[p1],self[p2])
            if self._rng.uniform() <= self._mp:
                self[n]=self._mutf(self[n], self._rng)
            if not self._ceval:
                self[n].score = self._scoref(self[n])
        if self._ceval:
            part_scores=self._scoref(self[nsurv:])
            for i,ind in enumerate(self[nsurv:]):
                ind.score = part_scores[i]
        self.sort()
    def run(self, n_iters, trj_fn="", verbose=0):
        if trj_fn:
            with open(trj_fn,'w') as f:
                for i in range(n_iters):
                    try:
                        self._iterate()
                    except ValueError:
                        print("Iterations Stoped due to population stagnation.")
                        break
        else:
            for i in range(n_iters):
                try:
                    self._iterate()
                except ValueError:
                    print("Iterations Stoped due to population stagnation.")
                    break
                if verbose==1:
                    print(f"Iter= {i}, best_score={self[0].score}, pop_std={np.std([i.score for i in self])}")
    def sort(self):
        list.sort(self, key= lambda x: x.score)

def mut_float_genes(i, rng):
    mut = np.ones(len(i))
    mut[rng.integers(len(i))] *= rng.normal()
    return Individual(np.array(i)*mut)

def cross_float_genes(i1, i2):
    return Individual(list(0.5*(np.array(i1)+np.array(i2))))


