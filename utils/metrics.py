def compare_counter(c1,c2):
    '''
    number of matches between two counters
    '''
    num_matches = 0
    for k,v in c1.items():
        if k in c2: 
            v2 = c2[k]
            num_matches += min(v2,v)
    return num_matches

def ngram(text, n):
    '''
    ngram
    '''
    ngrams = []
    for idx in range(len(text.split())-n+1):
        ngram_ = []
        for i in range(n):
            ngram_.append(text.split()[idx+i])
        ngrams.append(' '.join(ngram_))
    return ngrams

def rouge_n(gros,tars,n):
    '''
    rouge n-gram
    '''
    rouge_n_recall_o = 0
    rouge_n_precision_o = 0
    for idx in range(len(gros)):
        
        g,o = ngram(gros[idx],n),ngram(tars[idx],n)
        cg = Counter(g)
        co = Counter(o)
        n_matches_o  = compare_counter(cg,co)
        
        rouge_n_recall_o    += (n_matches_o)/len(g)
        rouge_n_precision_o += (n_matches_o)/len(o)
        
    rouge_n_f1_o        = 2*rouge_n_recall_o*rouge_n_precision_o/(rouge_n_precision_o+rouge_n_recall_o+0.0001)
        
    return rouge_n_recall_o/len(gros), rouge_n_precision_o/len(gros), rouge_n_f1_o/len(gros)


def LCS(text1, text2):
    '''
    Longest common subsequence between two list of words
    '''
    w1 = text1.split()
    w2 = text2.split()
    
    l,r = 0,0
    dp = {}
    
    def dfs(l,r):
        
        if l>=len(w1) or r >= len(w2):
            return 0
        
        if (l,r) in dp:
            return dp[(l,r)]
        
        if w1[l] == w2[r]:
            dp[(l,r)] = 1 + dfs(l+1,r+1)
        else: 
            dp[(l,r)] = max(dfs(l,r+1),dfs(l+1,r))
        
        return dp[(l,r)]
    
    return dfs(0,0)
    
def rouge_L(gros,tars):
    '''
    rouge LCS
    '''
    rougeL_f1_o = 0
    for idx in range(len(gros)):
        
        g,o = (gros[idx]),(tars[idx])
        lcs_matches  = LCS(g,o)
        
        rougeL_recall_o    = (lcs_matches)/len(g.split())
        rougeL_precision_o = (lcs_matches)/len(o.split())
        
        rougeL_f1_o        += 2*rougeL_recall_o*rougeL_precision_o/(rougeL_precision_o+rougeL_recall_o+0.0001)
        
    return rougeL_recall_o/len(gros),rougeL_precision_o/len(gros), rougeL_f1_o/len(gros)

def bleu(gros, tars, n):
    '''
    avg precision across range of ngrams
    '''
    pr_total = 0
    for i in range(1,n+1):
        _,pr_i,_ = rouge_n(gros,tars,i)
        pr_total += pr_i 
    return pr_total/n

def kl_divergence(p, q):
     return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
    
if __name__ == '__main__'
    # unit tests

    text1 = 'it is cold outside'
    text2 = 'it is too cold in the outside'

    assert LCS(text1,text2) == 4 , "LCS is not correct"


    text1 = ['it is dark']
    text2 = ['dark is it']

    assert rouge_n(text1,text2,1)[-1] >= 0.99 , "rouge_1 score don't match"
    assert rouge_n(text1,text2,2)[-1] <= 0.49 , "rouge_2 score don't match"
    assert rouge_L(text1,text2)[-1] <= 0.49 , "rouge_L score don't match"
    assert bleu(text1,text2,2) <= 0.5 , "bleu score don't match"
    assert bleu(text1,text2,2) ==  (rouge_n(text1,text2,1)[1] + rouge_n(text1,text2,2)[1])/2, "bleu is not the average of first two-gram prcisions"



    text1 = ['it is cold']
    text2 = ['it is very cold']

    assert rouge_n(text1,text2,1)[-1] >= 0.8 , "rouge_1 score don't match"
    assert rouge_n(text1,text2,2)[-1] >= 0.35 , "rouge_2 score don't match"
    assert rouge_L(text1,text2)[-1] >= 0.8 , "rouge_L score don't match"
    assert bleu(text1,text2,2) <= 0.6 , "bleu score don't match"
    assert bleu(text1,text2,2) ==  (rouge_n(text1,text2,1)[1] + rouge_n(text1,text2,2)[1])/2, "bleu is not the average of first two-gram prcisions"


