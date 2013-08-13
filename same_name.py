import pdb
from collections import defaultdict
import csv
import re
import nltk
import jellyfish
import sys

punctuations  = re.compile(r'[\-\@_.!,:;()]')
author_list = []
author_id_to_name = {}
auth_affiliation = defaultdict(lambda:[])
def get_signature_from_tokens(tokens):
    return ''.join([t[0].lower() for t in tokens if t!=''])
def get_author_clusters(auth):
    author_fmap = defaultdict(lambda:[])
    for i,row in enumerate(auth):
        if i==0:
            continue
        author_list.append(row[0])
        author_id_to_name[row[0]] = row[1]
        text = row[1]
        text =text.decode('ascii',errors='ignore')
        mod_text = punctuations.sub(r' ',text)
        tokens = mod_text.split(' ')
        sig = get_signature_from_tokens(tokens)
        #if 'Byung-Chai' in tokens:
        #    pdb.set_trace()
        if sig == '':
            continue
        author_fmap[sig].append((row[1],row[0]))
        auth_affiliation[row[0]] = row[2]
    return author_fmap

def get_tokens(s):
    text =s.decode('ascii',errors='ignore')
    mod_text = punctuations.sub(r' ',text)
    mod_text = mod_text.split(' ')
    mod_text = [h.lower() for h in mod_text if h!='']
    return sorted(mod_text)

def remove_similar_sounds(l,r):
    l_sig = []
    r_sig = []
    l_map = defaultdict(lambda:[])
    r_map = defaultdict(lambda:[])

    if len(l)!=len(r):
        pdb.set_trace()
    for tok in l:
        sig = jellyfish.soundex(tok)
        l_sig.append(sig)
        l_map[sig].append(tok)
    for tok in r:
        sig = jellyfish.soundex(tok)
        if sig in l_sig:
            l_sig.remove(sig)
        else:
            r_sig.append(sig)
            r_map[sig].append(tok)
    new_l = []
    for item in l_sig:
        for i in l_map[item]:
            new_l.append(i)
    new_r = []
    for item in r_sig:
        for i in r_map[item]:
            new_r.append(i)

    if len(new_l)!=len(new_r):
        pdb.set_trace()
    return (sorted(new_l),sorted(new_r))
    
def compare_tokenset(l,r):
    if l == r:
        return True
    else:
        return False
    #if the names sound similar, return true
    #if 'Byung' in l:
    #    pdb.set_trace()
    #(l, r) = remove_similar_sounds(l,r)
    length = len(l)
    #if 'VAZIRANIz' in l:
    #    pdb.set_trace()
    for i in range(length):
        try:
            if (len(l[i]) == 1 or len(r[i])==1) and (l[i][0] == r[i][0]):
                continue
        except:
            pdb.set_trace()
        if (len(l[i]) > 1 and len(r[i]) > 1) and (nltk.edit_distance(l[i],r[i]) <= 2) and len(l[i])>4:
            continue
        else:
            return False
    return True
def tok_cmp(x,y):
    if x[0]>y[0]:
        return 1
    if x[0]<y[0]:
        return -1
    else:
        return 0

author_duplicates = defaultdict(lambda:[])
def create_groups_within_clusters(clusters):
    pdb.set_trace()
    for val in clusters.values():
        val = sorted(val,cmp=tok_cmp)
        tokenset = {}
        for v in val:
            tokens = get_tokens(v[0])
            if tokenset.has_key(v[0])==False:
                tokenset[v[0]] = tokens

        l = len(val)
        clust = []
        prev = [val[0]]
        
        for i in range(l):
            if i+1!=l:
                left = tokenset[val[i][0]]
                right = tokenset[val[i+1][0]]
                ret = compare_tokenset(left,right)
                if ret == True:
                    prev.extend([val[i+1]])
                else:
                    clust.append(prev)
                    prev = [val[i+1]]
        clust.append(prev)
        for cl in clust:
#            print cl
            for c in cl:
                author_duplicates[c[1]].extend(cl)
 #               print " affiliation :"+auth_affiliation[c[1]]
        

if __name__=='__main__':
    auth = csv.reader(open('Author.csv','r'))
    print "mapping author ids to signatures"
    auth_clusters = get_author_clusters(auth) 
    #pdb.set_trace()
    create_groups_within_clusters(auth_clusters)
    author_map = defaultdict(lambda:[])
    for author in author_list:
        if author_duplicates[author] != []:
            if sys.argv[2] == 'text':
                author_map[author].extend([v[0] for v in author_duplicates[author]])
            else:
                author_map[author].extend([v[1] for v in author_duplicates[author]])
        else:
            author_map[author] = [author]
    pdb.set_trace()
    fh = open(sys.argv[1]+sys.argv[2],'w')
    fh.write("AuthorId,DuplicateAuthorIds\n")
    for k,v in author_map.items():
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


