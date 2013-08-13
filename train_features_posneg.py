import pdb
from collections import defaultdict
import csv
import re
import nltk
import jellyfish
import sys
from UnionFind import UnionFind
import pickle
import codecs
import numpy as np
import gensim

import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from unidecode import unidecode
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation



uf = UnionFind()
punctuations  = re.compile(r'[\-\@_.!,:;()]')
author_list = []
author_id_to_name = {}
author_id_to_normname = {}
auth_affiliation = defaultdict(lambda:[])
authors_per_paper = defaultdict(lambda:set())
paper_table_map = {}

pos_classifier = svm.SVC(kernel='linear',  C=0.7, probability=True)
neg_classifier = svm.SVC(kernel='linear',  C=0.7, probability=True)

instance_count = 0
def get_signature_from_tokens(tokens):
    return ''.join([t[0].lower() for t in tokens if t!=''])
def get_normname(tokens):
    t = [tok.lower() for tok in tokens if tok!='']
    return ' '.join(t)
def get_signature(text):
    text =text.decode('ascii',errors='ignore')
    mod_text = punctuations.sub(r' ',text)
    tokens = mod_text.split(' ')
    sig = get_signature_from_tokens(tokens)
    if sig =='':
        return 'NONESIG'
    else:
        return sig

import nltk
stopwords = nltk.corpus.stopwords.words()
stopwords_map = {word:1 for word in stopwords}
def get_tokens(s):
    text =s.decode('ascii',errors='ignore')
    mod_text = punctuations.sub(r' ',text)
    mod_text = mod_text.split(' ')
    mod_text = [h.lower() for h in mod_text if h!='' and not stopwords_map.has_key(h)]
    return mod_text

def get_last_name(s):
    mod_text = punctuations.sub(r' ',s)
    mod_text = mod_text.split(' ')
    mod_text = [h.lower() for h in mod_text if h!='']
    if mod_text!=[]:
        return mod_text[-1]
    else:
        return None


from gensim import corpora, models, similarities
from gensim.similarities import SparseMatrixSimilarity,Similarity
import numpy

def read_author(auth):
    for i,row in enumerate(auth):
        if i==0:
            continue
        author_list.append(row[0])
        author_id_to_name[row[0]] = row[1]
        text = row[1]
        text = unidecode(text.decode('utf-8')).encode('utf-8')
        #text =text.decode('ascii',errors='ignore')
        mod_text = punctuations.sub(r' ',text)
        tokens = [h for h in mod_text.split(' ') if h!='']
        #sig = get_signature_from_tokens(tokens)
        #if 'Byung-Chai' in tokens:
        #    pdb.set_trace()
        #if sig == '':
        #    continue
        normname = get_normname(tokens)
        author_id_to_normname[row[0]] = [normname, tokens]
        auth_affiliation[row[0]] = get_tokens(row[2])
 

author_train_confirmed = defaultdict(lambda:set())
author_train_deleted = defaultdict(lambda:set())
def read_train(auth):
    for i,row in enumerate(auth):
        if i==0:
            continue
        author_train_confirmed[row[0]] = set(row[1].split(' '))
        #author_train_deleted[row[0]] = set(row[2].split(' '))
 

def get_last_name(text):
    return text.split(' ')[-1]

last_name_clusters = defaultdict(lambda:[])
def get_last_name_clusters(last_name_clusters):
    for k,v in author_id_to_normname.items():
        last_name_clusters[get_last_name(v[0])].append((k,v[0],v[1]))

def fgen_signature(x,y):
    sig1 = get_signature(x[1])
    sig2 = get_signature(y[1])
    if sig1 == sig2:
        return 0.0
    else:
        return -1.0

def get_abstract_score(text):
    tokens = text.split(' ')
    score=len(tokens)
    for tok in tokens:
        if len(tok)==1:
            score -=1
    return float(score)/len(tokens)

from gensim import corpora, models, similarities
#positive feature
def fgen_paper_table(x,y):
    if paper_table_map.has_key(x[0]) and paper_table_map.has_key(y[0]):
        left = set(paper_table_map[x[0]])
        right = set(paper_table_map[y[0]])
        
        if len(left & right)>0:
            res = left&right
            absscore = [get_abstract_score(r) for r in res]
            return max(absscore)
    return 0.0

#positive feature
def fgen_affiliation(x,y):
    try:
        l = auth_affiliation_collated[x[0]]
        r = auth_affiliation_collated[y[0]]
    except:
        pdb.set_trace()
    if l != [] and r!=[]:
        lt  = set(l)
        rt  = set(r)
        try:
            return float(len(lt&rt))/len(lt|rt)
        except:
            pdb.set_trace()
    return 0.0


#between +ve and negative ( more on the positive)
def fgen_normname_tokens(x,y):
    l = x[1].split(' ')
    r = y[1].split(' ')
    sig1 = get_signature_from_tokens(x[2])
    sig2 = get_signature_from_tokens(y[2])
    if len(sig1) == len(sig2):
        if sig1!=sig2:
            return -1.0
    l.pop()
    r.pop()
    dist = jellyfish.jaro_distance(' '.join(l),' '.join(r))
    return dist
   
def fgen_train(x,y):
    confirmed1 = author_train_confirmed[x[0]]
    #deleted1 = author_train_deleted[x[0]]
    confirmed2 = author_train_confirmed[y[0]]
    #deleted2 = author_train_deleted[y[0]]
    #if len(confirmed2 & deleted1)+len(confirmed1 & deleted2) > 0:
    #    pdb.set_trace()
    #    return -1.0
    if len(confirmed2 & confirmed1) > 0:
        pdb.set_trace
        return 1.0
    else:
        return 0.5


#negative feature

def fgen_papers(x,y):
    if papers_per_author.has_key(x[0]) and papers_per_author.has_key(y[0]):
        papers1 = set(papers_per_author[x[0]])
        papers2 = set(papers_per_author[y[0]])
        l = len(papers1 & papers2)
        if l>0:
            #return float(l)/len(papers1|papers2) 
            return -1.0 
    return 0.0

#positive feature
def fgen_samename(x,y):
    if x[1] == y[1]:
        toks1 = x[2]
        toks2 = y[2]
        if len(toks1[0]) == 1: 
            return 0.0
        return 1.0
    else:
        return 0.0
def compare_tokens(l,r):
    if r.startswith(l) or l.startswith(r):
        return 1.0
    if l!=r:
        return -1.0
def compare_midnames(l,r):
    try:
        l.pop(-1)
        r.pop(-1)
    except:
        pass

    l = sorted(l[1:]) 
    r = sorted(r[1:])
    if len(l)<len(r):
        small = l
        large = r
    else:
        small = r
        large = l

    i = 0
    j = 0
    while i!=len(small) and j!=len(large):
        if small[i][0] == large[j][0] and (small[i].startswith(large[j]) or large[j].startswith(small[i])):
            i +=1
            j +=1
        elif small[i][0] == large[j][0] and small[i]!=large[j]:
            return -1.0
        #if small[i][0] < large[j][0] or small[:
        else:
            j +=1
    if i!=len(small) and j==len(large):
        return -1.0

    return 1.0 

   
#negative feature
   
def fgen_tokens(x,y):
    l = [u.lower() for u in x[2]]
    r = [u.lower() for u in y[2]]

    if l[0] == r[0]:
        return compare_midnames(l,r)
    if len(l[0]) == 1 and r[0].startswith(l[0]) or len(r[0]) == 1 and l[0].startswith(r[0]):
        return compare_midnames(l,r)
    
    if len(l) == len(r):
        if r[0].startswith(l[0]):
            #we are good with the first name,lets explore the other tokens
            for i in range(1,len(l)):
                h= compare_tokens(l[i],r[i])
                if h<0:
                    return -1.0
            return 1.0
    #if len(l[0])>1 and r[0].startswith(l[0]):
    #   return compare_midnames(l,r)
    return -1.0
    

#positive feature
def fgen_coauthor(x,y):
    if coauths.has_key(x[0]) and coauths.has_key(y[0]):
        c1 = coauths[x[0]]
        c2 = coauths[y[0]]
    #if y[0] in c1 or x[0] in c2:
    #    pdb.set_trace()
    #    return -1.0
        l = len(c1 & c2)
        if l>0:
            res = float(l)/len(c1|c2) 
            if res == 0:
                return -1.0
            else:
                return res
    return 0.0


     
fgen_array = [fgen_affiliation, fgen_samename, fgen_paper_table,  fgen_normname_tokens,fgen_papers,fgen_tokens,fgen_coauthor]
fgen_pos_array = [fgen_affiliation, fgen_samename, fgen_paper_table,fgen_coauthor, fgen_normname_tokens]
fgen_neg_array = [fgen_papers, fgen_tokens, fgen_coauthor]

train_list = []
def gen_features(obj1, obj2, fgen_arr):
    global instance_count
    scores_arr = []
    for i in range(len(fgen_arr)):
        scores_arr.append(fgen_arr[i](obj1,obj2))
    scoresvec = np.array(scores_arr)
    #result = np.dot(weightvec,scoresvec)
    #if flag == 1 and obj1[1]=='david l elliott':
    #    pdb.set_trace()
    #    if result >0.9:
    #        print obj1, obj2, result,scoresvec
    #        pdb.set_trace()
    #if result>0.9:
        #print obj1, obj2, result,scoresvec
        #label = float(raw_input())
        #train_X[instance_count] = scoresvec
        #train_Y[instance_count] = label
        #if instance_count == 99:
        #    print "training done"
        #pdb.set_trace()
        #instance_count +=1
        #train_list.append([obj1, obj2, result, scoresvec])
    return scoresvec

def val_cmp(x,y):
    if x[1]>y[1]:
        return 1
    if x[1]<y[1]:
        return -1
    else:
        return 0

def allpairs_cmp(x,y):
    pdb.set_trace()
def get_instances_and_train(fgen_arr,classifier):
    f = open('pos.tr.data','r')
    lines = f.read().splitlines()
    len_pos = len(lines) 
    X_train = np.zeros((30000,len(fgen_arr)))
    Y_train = np.zeros((30000,1))
    print "training positive instances %d"%len_pos
    for i,line in enumerate(lines):
        cols = line.split(' ')
        auth1 = [cols[0]]
        auth2 = [cols[1]]
        auth1.extend(author_id_to_normname[cols[0]])
        auth2.extend(author_id_to_normname[cols[1]])
        scoresvec = gen_features(auth1,auth2,fgen_arr)
        label = int(cols[2])
        X_train[i] = scoresvec
        Y_train[i] = label
    f = open('neg.tr.data','r')
    lines = f.read().splitlines()
    len_neg = 30000-len_pos
    print "training negative instances %d"%len_neg
    for i,line in enumerate(lines[:len_neg]):
        cols = line.split(' ')
        auth1 = [cols[0]]
        auth2 = [cols[1]]
        auth1.extend(author_id_to_normname[cols[0]])
        auth2.extend(author_id_to_normname[cols[1]])
        scoresvec = gen_features(auth1,auth2,fgen_arr)
        label = int(cols[2])
        try:
            X_train[i+len_pos] = scoresvec
            Y_train[i+len_pos] = label
        except:
            pdb.set_trace()


    classifier = classifier.fit(X_train, Y_train)
    print "done training"
    #pickle.dump(classifier, open('model1','w'))
    #probas_ = classifier.predict_proba(X_train)
    #fpr, tpr, thresholds = roc_curve(Y_train, probas_[:, 1])
    #roc_auc = auc(fpr, tpr)
    #print "Area under the ROC curve : %f" % roc_auc
    #X_test = np.zeros((10000,7))
    #Y_test = np.zeros((10000,1))
    #for i,line in enumerate(lines[len_neg:len_neg+10000]):
    #    cols = line.split(' ')
    #    auth1 = [cols[0]]
    #    auth2 = [cols[1]]
    #    auth1.extend(author_id_to_normname[cols[0]])
    #    auth2.extend(author_id_to_normname[cols[1]])
    #    scoresvec = gen_features(auth1,auth2)
    #    X_test[i] = scoresvec
    #    label = int(cols[2])
    #    Y_test[i] = label 
    #probas_ = classifier.predict_proba(X_test)
    #fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
    #roc_auc = auc(fpr, tpr)
    #print "Area under the ROC curve : %f" % roc_auc
   
    #scores = cross_validation.cross_val_score(classifier, X_train, Y_train, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    # Plot ROC curve
    #pl.clf()
    #pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    #pl.plot([0, 1], [0, 1], 'k--')
    #pl.xlim([0.0, 1.0])
    #pl.ylim([0.0, 1.0])
    #pl.xlabel('False Positive Rate')
    #pl.ylabel('True Positive Rate')
    #pl.title('Receiver operating characteristic example')
    #pl.legend(loc="lower right")
    #pl.show()
        
    
    
def len_cmp(x,y):
    if len(x[1]) > len(y[1]):
        return -1
    if len(x[1]) < len(y[1]):
        return 1
    else:
        return 0

def generate_features(clusters):
    count = 0
    print len(clusters)
    clusters = sorted(clusters.items(),cmp=len_cmp)
    ncl_real = []
    for key,val in clusters:
        l = len(val)
        #TODO: handle "" clusters and single element clusters, "vaziraniz"
        val = sorted(val,cmp=val_cmp)
        #uf.insert_objects([c[0] for c in val]) 
        print "cluster: "+key+" nelems:%d"%len(val)
        count += l*l
        if l==1:
            ncl_real.append([val[0][0]])
            continue
        if key == '':
            ncl_real.extend([[c[0]] for c in val])
            continue

        graph = nx.Graph()
         
        node_map = {}
        for c in val:
            node_map[c[0]] = c

        graph.add_nodes_from([c[0] for c in val])
        for i in range(l):
            for j in range(i+1,l):
                scoresvec = gen_features(val[i],val[j],fgen_pos_array)
                res = pos_classifier.predict(scoresvec)
                if res[0]>0.0:
                    #print "joining %s,%s"%(val[i][1],val[j][1]) 
                    graph.add_edges_from([(val[i][0],val[j][0])],weight=res)
                    #graph.add_edges_from([(val[i],val[j])],weight=res)
        
        #cli = clique_removal(graph)
        cli = nx.connected_components(graph)
        #pdb.set_trace()
        #for c in cli[1]:
        #    h = []
        #    for m in c:
        #        h.append(m)
        #    ncl_real.append(h)

        new_real = []
        for cl in cli:
            val = [node_map[c] for c in cl]    
            l = len(val)
            val = sorted(val,cmp=val_cmp)
            #print "cluster: "+key+" nelems:%d"%len(val)

            if l==1:
                new_real.append([val[0][0]])
                continue
            #if key == '':
            #    new_real.append([val[0][0]])
            #    continue

            graph = nx.Graph()
             
            graph.add_nodes_from([c[0] for c in val])
            for i in range(l):
                for j in range(i+1,l):
                    scoresvec = gen_features(val[i],val[j],fgen_neg_array)
                    res = neg_classifier.predict(scoresvec)
                    #print "looking at %s,%s %f"%(val[i][1],val[j][1],res) 
                    #print scoresvec
                    if res > 0.0:
                        #print "making %s,%s"%(val[i][1],val[j][1]) 
                        graph.add_edges_from([(val[i][0],val[j][0])],weight=res)
                        #graph.add_edges_from([(val[i],val[j])],weight=res)
            
            #pdb.set_trace()
            cli = clique_removal(graph)
            for c in cli[1]:
                h = []
                for m in c:
                    h.append(m)
                new_real.append(h)
        ncl_real.extend(new_real)

    return ncl_real

            
def write_aff_map(aff_map):
    f = open('affinity_collated.csv','w')
    row_map = {}
    for k,v in author_id_to_name.items():
        row_map[k] = [v, author_id_to_normname[k][0]]
    if aff_map.has_key(k):
        row_map[k].append(':'.join(aff_map[k]))
    else:
        row_map[k].append('')
    if paper_table_map.has_key(k):
        row_map[k].append(':'.join(paper_table_map[k]))
    else:
        row_map[k].append('')


    writer = csv.writer(f)
    for k,v in row_map.items():
        row = [k]
        row.extend(v)
        writer.writerow(row)
    f.close()
    pdb.set_trace() 

    
auth_affiliation_collated = defaultdict(lambda:[])    
def process_aff_map(aff_map):
    for k,v in aff_map.items():
        for val in v:
            auth_affiliation_collated[k].extend(get_tokens(val))


def heu_cmp(x,y):
    pass

def process_further(clusters):
    for cluster in clusters:
        names = [author_id_to_normname[h] for h in cluster]
        names_sorted = sorted(names,cmp=heu_cmp)




def read_NormPaperAuth(paperAuth):
    for row in paperAuth:
        r = row[0].split('\t')
        authors_per_paper[r[0]] = set(r[1].split())


if __name__=='__main__':
    auth = csv.reader(open('Author.csv','r'))
    train = csv.reader(open('Valid.csv','r'))
    print "hang by last names"
    read_author(auth) 
    aff_map = pickle.load(open('aff_map.pickle','r'))
    coauths = pickle.load(open('coauthors.pickle','r'))
    #read_train(train) 
    get_last_name_clusters(last_name_clusters)
    paper_table_map = pickle.load(open('token_map_paperauth_aff_map_1.pickle','r'))
    papers_per_author = pickle.load(open('papers_per_author.pickle','r'))
    process_aff_map(aff_map)
    #write_aff_map(aff_map)
    get_instances_and_train(fgen_pos_array,pos_classifier)
    get_instances_and_train(fgen_neg_array,neg_classifier)
    ncl_real = generate_features(last_name_clusters)
    #process_further(uf.get_clusters())
    #same_name_clusters = uf.get_clusters()
    #cluster_list = []
    #cl_real = [h for h in same_name_clusters.values() if len(h)!=0]
    cl_real = ncl_real
    f = open(sys.argv[1],'w')
    for cl in cl_real:
        cl_s = [author_id_to_name[c] for c in cl]
        f.write(str(cl_s)+'\n')
    f.close()

    fh = open(sys.argv[1]+sys.argv[2],'w')
    fh.write("AuthorId,DuplicateAuthorIds\n")
    for cl in cl_real:
        auth_map = {}
        for c in cl:
            auth_map[c] = cl
        for k,v in auth_map.items():
            if sys.argv[2] == 'text':
                k = author_id_to_name[k]
            fh.write("%s,%s"%(k,k))
            for val in v:
                if val == k:
                    continue
                if sys.argv[2] == 'text':
                    fh.write('||')
                else:
                    fh.write(' ')
                fh.write("%s"%val)
            fh.write("\n")
    fh.close()
 
