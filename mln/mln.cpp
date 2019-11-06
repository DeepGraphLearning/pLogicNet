#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>

#define MAX_STRING 1000
#define MAX_THREADS 100

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double inv_sigmoid(double x)
{
    return -log(1.0 / x - 1.0);
}

int min(int a, int b)
{
    if (a < b) return a;
    return b;
}

struct Triplet
{
    int h, t, r;
    char type;
    int valid;
    double truth, logit;
    std::vector<int> rule_ids;
    
    Triplet()
    {
        h = -1;
        t = -1;
        r = -1;
        type = -1;
        valid = -1;
        truth = 0;
        logit = 0;
        rule_ids.clear();
    }
    
    ~Triplet()
    {
        rule_ids.clear();
    }
    
    void init()
    {
        truth = 0;
        logit = 0;
        rule_ids.clear();
    }
    
    friend bool operator < (Triplet u, Triplet v)
    {
        if (u.r == v.r)
        {
            if (u.h == v.h) return u.t < v.t;
            return u.h < v.h;
        }
        return u.r < v.r;
    }
    
    friend bool operator == (Triplet u, Triplet v)
    {
        if (u.h == v.h && u.t == v.t && u.r == v.r) return true;
        return false;
    }
};

struct Pair
{
    int e, r;
};

struct Rule
{
    std::vector<int> r_premise;
    int r_hypothesis;
    std::string type;
    double precision, weight, grad;
    
    Rule()
    {
        precision = 0;
        weight = 0;
        grad = 0;
    }
    
    friend bool operator < (Rule u, Rule v)
    {
        if (u.type == v.type)
        {
            if (u.r_hypothesis == v.r_hypothesis)
            {
                int min_length = min(int(u.r_premise.size()), int(v.r_premise.size()));
                for (int k = 0; k != min_length; k++)
                {
                    if (u.r_premise[k] != v.r_premise[k])
                    return u.r_premise[k] < v.r_premise[k];
                }
            }
            return u.r_hypothesis < v.r_hypothesis;
        }
        return u.type < v.type;
    }
};

char observed_triplet_file[MAX_STRING], probability_file[MAX_STRING], output_rule_file[MAX_STRING], output_prediction_file[MAX_STRING], output_hidden_file[MAX_STRING], save_file[MAX_STRING], load_file[MAX_STRING];
int entity_size = 0, relation_size = 0, triplet_size = 0, observed_triplet_size = 0, hidden_triplet_size = 0, rule_size = 0, iterations = 10, num_threads = 1;
double rule_threshold = 0, triplet_threshold = 1, learning_rate = 0.01;
long long total_count = 0;
std::map<std::string, int> ent2id, rel2id;
std::vector<std::string> id2ent, id2rel;
std::vector<Triplet> triplets;
std::vector<Pair> *h2rt = NULL;
std::set<Rule> candidate_rules;
std::vector<Rule> rules;
std::set<Triplet> observed_triplets, hidden_triplets;
std::map<Triplet, double> triplet2prob;
std::map<Triplet, int> triplet2id;
std::vector<int> rand_idx;
sem_t mutex;

/* Debug */
void print_rule(Rule rule)
{
    for (int k = 0; k != int(rule.r_premise.size()); k++) printf("%s ", id2rel[rule.r_premise[k]].c_str());
    printf("-> %s %s\n", id2rel[rule.r_hypothesis].c_str(), rule.type.c_str());
}

/* Debug */
void print_triplet(Triplet triplet)
{
    int h = triplet.h;
    int t = triplet.t;
    int r = triplet.r;
    printf("%s %s %s\n", id2ent[h].c_str(), id2rel[r].c_str(), id2ent[t].c_str());
    printf("%c %d %lf %lf\n", triplet.type, triplet.valid, triplet.truth, triplet.logit);
    for (int k = 0; k != int(triplet.rule_ids.size()); k++) print_rule(rules[triplet.rule_ids[k]]);
    printf("\n");
    printf("\n");
}

void read_data()
{
    char s_head[MAX_STRING], s_tail[MAX_STRING], s_rel[MAX_STRING];
    int h, t, r;
    Triplet triplet;
    Pair ent_rel_pair;
    std::map<std::string, int>::iterator iter;
    FILE *fi;
    
    fi = fopen(observed_triplet_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: file of observed triplets not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%s %s %s", s_head, s_rel, s_tail) != 3) break;
        
        if (ent2id.count(s_head) == 0)
        {
            ent2id[s_head] = entity_size;
            id2ent.push_back(s_head);
            entity_size += 1;
        }
        
        if (ent2id.count(s_tail) == 0)
        {
            ent2id[s_tail] = entity_size;
            id2ent.push_back(s_tail);
            entity_size += 1;
        }
        
        if (rel2id.count(s_rel) == 0)
        {
            rel2id[s_rel] = relation_size;
            id2rel.push_back(s_rel);
            relation_size += 1;
        }
        
        h = ent2id[s_head]; t = ent2id[s_tail]; r = rel2id[s_rel];
        triplet.h = h; triplet.t = t; triplet.r = r; triplet.type = 'o'; triplet.valid = 1;
        triplets.push_back(triplet);
        observed_triplets.insert(triplet);
    }
    fclose(fi);
    
    observed_triplet_size = int(triplets.size());
    
    h2rt = new std::vector<Pair> [entity_size];
    for (int k = 0; k != observed_triplet_size; k++)
    {
        h = triplets[k].h; r = triplets[k].r; t = triplets[k].t;
        
        ent_rel_pair.e = t;
        ent_rel_pair.r = r;
        h2rt[h].push_back(ent_rel_pair);
    }
    
    printf("#Entities: %d          \n", entity_size);
    printf("#Relations: %d          \n", relation_size);
    printf("#Observed triplets: %d          \n", observed_triplet_size);
}

bool check_observed(Triplet triplet)
{
    if (observed_triplets.count(triplet) != 0) return true;
    else return false;
}

void search_composition_rules(int h, int r, int t)
{
    int len0, len1, mid, r0, r1;
    Rule rule;
    
    len0 = int(h2rt[h].size());
    for (int k = 0; k != len0; k++)
    {
        mid = h2rt[h][k].e;
        r0 = h2rt[h][k].r;
        
        len1 = int(h2rt[mid].size());
        for (int i = 0; i != len1; i++)
        {
            if (h2rt[mid][i].e != t) continue;
            
            r1 = h2rt[mid][i].r;
            
            rule.r_premise.clear();
            rule.r_premise.push_back(r0);
            rule.r_premise.push_back(r1);
            rule.r_hypothesis = r;
            rule.type = "composition";
            candidate_rules.insert(rule);
        }
    }
}

void search_symmetric_rules(int h, int r, int t)
{
    int len;
    Rule rule;
    
    len = int(h2rt[t].size());
    for (int k = 0; k != len; k++)
    {
        if (h2rt[t][k].r != r) continue;
        if (h2rt[t][k].e != h) continue;
        
        rule.r_premise.clear();
        rule.r_premise.push_back(r);
        rule.r_hypothesis = r;
        rule.type = "symmetric";
        candidate_rules.insert(rule);
    }
    
}

void search_inverse_rules(int h, int r, int t)
{
    int len, invr;
    Rule rule;
    
    len = int(h2rt[t].size());
    for (int k = 0; k != len; k++)
    {
        if (h2rt[t][k].r == r) continue;
        if (h2rt[t][k].e != h) continue;
        
        invr = h2rt[t][k].r;
        
        rule.r_premise.clear();
        rule.r_premise.push_back(invr);
        rule.r_hypothesis = r;
        rule.type = "inverse";
        candidate_rules.insert(rule);
    }
}

void search_subrelation_rules(int h, int r, int t)
{
    int len, subr;
    Rule rule;
    
    len = int(h2rt[h].size());
    for (int k = 0; k != len; k++)
    {
        if (h2rt[h][k].e != t) continue;
        
        subr = h2rt[h][k].r;
        if (subr == r) continue;
        
        rule.r_premise.clear();
        rule.r_premise.push_back(subr);
        rule.r_hypothesis = r;
        rule.type = "subrelation";
        candidate_rules.insert(rule);
    }
}

void search_candidate_rules()
{
    for (int k = 0; k != observed_triplet_size; k++)
    {
        if (k % 100 == 0)
        {
            printf("Progress: %.3lf%%          %c", (double)k / (double)(observed_triplet_size + 1) * 100, 13);
            fflush(stdout);
        }

        search_composition_rules(triplets[k].h, triplets[k].r, triplets[k].t);
        search_symmetric_rules(triplets[k].h, triplets[k].r, triplets[k].t);
        search_inverse_rules(triplets[k].h, triplets[k].r, triplets[k].t);
        search_subrelation_rules(triplets[k].h, triplets[k].r, triplets[k].t);
    }
    
    std::set<Rule>::iterator iter;
    for (iter = candidate_rules.begin(); iter != candidate_rules.end(); iter++)
    rules.push_back(*iter);
    
    rule_size = int(candidate_rules.size());
    candidate_rules.clear();
    printf("#Candidate rules: %d          \n", rule_size);

    for (int k = 0; k != rule_size; k++) rand_idx.push_back(k);
    std::random_shuffle(rand_idx.begin(), rand_idx.end());
}

double precision_composition_rule(Rule rule)
{
    int len, h, mid, t;
    int rp0, rp1, rh;
    double p = 0, q = 0;
    Triplet triplet;
    
    rp0 = rule.r_premise[0];
    rp1 = rule.r_premise[1];
    rh = rule.r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp0) continue;
        
        h = triplets[k].h;
        mid = triplets[k].t;
        
        for (int i = 0; i != int(h2rt[mid].size()); i++)
        {
            if (h2rt[mid][i].r != rp1) continue;
            
            t = h2rt[mid][i].e;
            triplet.h = h; triplet.r = rh; triplet.t = t;
            
            if (check_observed(triplet) == true) p += 1;
            q += 1;
        }
    }
    
    return p / q;
}

double precision_symmetric_rule(Rule rule)
{
    int h, t, rp, rh, len;
    double p = 0, q = 0;
    Triplet triplet;
    
    rp = rule.r_premise[0];
    rh = rule.r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (check_observed(triplet) == true) p += 1;
        q += 1;
    }
    
    return p / q;
}

double precision_inverse_rule(Rule rule)
{
    int h, t, rp, rh, len;
    double p = 0, q = 0;
    Triplet triplet;
    
    rp = rule.r_premise[0];
    rh = rule.r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (check_observed(triplet) == true) p += 1;
        q += 1;
    }
    
    return p / q;
}

double precision_subrelation_rule(Rule rule)
{
    int h, t, rp, rh, len;
    double p = 0, q = 0;
    Triplet triplet;
    
    rp = rule.r_premise[0];
    rh = rule.r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = h; triplet.t = t; triplet.r = rh;
        
        if (check_observed(triplet) == true) p += 1;
        q += 1;
    }
    
    return p / q;
}

void *compute_rule_precision_thread(void *id)
{
    int thread = int((long)(id));
    int bg = int(rule_size / num_threads) * thread;
    int ed = int(rule_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = rule_size;
    
    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)
        {
            total_count += 10;
            printf("Progress: %.3lf%%          %c", (double)total_count / (double)(rule_size + 1) * 100, 13);
            fflush(stdout);
        }

        int k = rand_idx[T];

        if (rules[k].type == "composition") rules[k].precision = precision_composition_rule(rules[k]);
        if (rules[k].type == "symmetric") rules[k].precision = precision_symmetric_rule(rules[k]);
        if (rules[k].type == "inverse") rules[k].precision = precision_inverse_rule(rules[k]);
        if (rules[k].type == "subrelation") rules[k].precision = precision_subrelation_rule(rules[k]);
    }
    
    pthread_exit(NULL);
}

void compute_rule_precision()
{
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    total_count = 0;
    for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, compute_rule_precision_thread, (void *)a);
    for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    free(pt);
    
    std::vector<Rule> rules_copy(rules);
    rules.clear();
    for (int k = 0; k != rule_size; k++)
    {
        if (rules_copy[k].precision >= rule_threshold) rules.push_back(rules_copy[k]);
    }
    rules_copy.clear();
    
    rule_size = int(rules.size());
    printf("#Final Rules: %d          \n", rule_size);
}

void search_hidden_with_composition(int id, int thread)
{
    int h, mid, t, len;
    int rp1, rp2, rh;
    Triplet triplet;
    
    rp1 = rules[id].r_premise[0];
    rp2 = rules[id].r_premise[1];
    rh = rules[id].r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp1) continue;
        
        h = triplets[k].h;
        mid = triplets[k].t;
        
        for (int i = 0; i != int(h2rt[mid].size()); i++)
        {
            if (h2rt[mid][i].r != rp2) continue;
            
            t = h2rt[mid][i].e;
            triplet.h = h; triplet.t = t; triplet.r = rh;
            
            if (check_observed(triplet) == true) continue;
            triplet.type = 'h'; triplet.valid = 0;
            sem_wait(&mutex);
            hidden_triplets.insert(triplet);
            sem_post(&mutex);
        }
    }
}

void search_hidden_with_symmetric(int id, int thread)
{
    int h, t, rp, rh, len;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (check_observed(triplet) == true) continue;
        triplet.type = 'h'; triplet.valid = 0;
        sem_wait(&mutex);
        hidden_triplets.insert(triplet);
        sem_post(&mutex);
    }
}

void search_hidden_with_inverse(int id, int thread)
{
    int h, t, rp, rh, len;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (check_observed(triplet) == true) continue;
        triplet.type = 'h'; triplet.valid = 0;
        sem_wait(&mutex);
        hidden_triplets.insert(triplet);
        sem_post(&mutex);
    }
}

void search_hidden_with_subrelation(int id, int thread)
{
    int h, t, rp, rh, len;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = h; triplet.t = t; triplet.r = rh;
        
        if (check_observed(triplet) == true) continue;
        triplet.type = 'h'; triplet.valid = 0;
        sem_wait(&mutex);
        hidden_triplets.insert(triplet);
        sem_post(&mutex);
    }
}

void *search_hidden_triplets_thread(void *id)
{
    int thread = int((long)(id));
    int bg = int(rule_size / num_threads) * thread;
    int ed = int(rule_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = rule_size;
    
    for (int k = bg; k != ed; k++)
    {
        if (k % 10 == 0)
        {
            total_count += 10;
            printf("Progress: %.3lf%%          %c", (double)total_count / (double)(rule_size + 1) * 100, 13);
            fflush(stdout);
        }

        if (rules[k].type == "composition") search_hidden_with_composition(k, thread);
        if (rules[k].type == "symmetric") search_hidden_with_symmetric(k, thread);
        if (rules[k].type == "inverse") search_hidden_with_inverse(k, thread);
        if (rules[k].type == "subrelation") search_hidden_with_subrelation(k, thread);
    }
    
    pthread_exit(NULL);
}

void search_hidden_triplets()
{
    sem_init(&mutex, 0, 1);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    total_count = 0;
    for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, search_hidden_triplets_thread, (void *)a);
    for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    free(pt);
    
    hidden_triplet_size = int(hidden_triplets.size());
    triplet_size = observed_triplet_size + hidden_triplet_size;
    
    std::set<Triplet>::iterator iter;
    for (iter = hidden_triplets.begin(); iter != hidden_triplets.end(); iter++) triplets.push_back(*iter);
    printf("#Hidden triplets: %d          \n", hidden_triplet_size);
    printf("#Triplets: %d          \n", triplet_size);
}

void read_probability_of_hidden_triplets()
{
    if (probability_file[0] == 0)
    {
        Pair ent_rel_pair;
        
        triplet2id.clear();
        for (int k = 0; k != entity_size; k++) h2rt[k].clear();
        
        for (int k = 0; k != triplet_size; k++)
        {
            triplet2id[triplets[k]] = k;
            
            if (triplets[k].type == 'o')
            {
                triplets[k].valid = 1;
                triplets[k].truth = 1;
                ent_rel_pair.e = triplets[k].t;
                ent_rel_pair.r = triplets[k].r;
                h2rt[triplets[k].h].push_back(ent_rel_pair);
            }
            else
            {
                triplets[k].valid = 0;
                triplets[k].truth = 0;
            }
        }
        return;
    }
    
    char s_head[MAX_STRING], s_tail[MAX_STRING], s_rel[MAX_STRING];
    double prob;
    Triplet triplet;
    
    FILE *fi = fopen(probability_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: probability file not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%s %s %s %lf", s_head, s_rel, s_tail, &prob) != 4) break;
        
        if (ent2id.count(s_head) == 0) continue;
        if (ent2id.count(s_tail) == 0) continue;
        if (rel2id.count(s_rel) == 0) continue;
        
        triplet.h = ent2id[s_head];
        triplet.t = ent2id[s_tail];
        triplet.r = rel2id[s_rel];
        
        triplet2prob[triplet] = prob;
    }
    fclose(fi);
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].type == 'o')
        {
            triplets[k].truth = 1;
            triplets[k].valid = 1;
            continue;
        }
        
        if (triplet2prob.count(triplets[k]) != 0 && triplet2prob[triplets[k]] >= triplet_threshold)
        {
            triplets[k].truth = triplet2prob[triplets[k]];
            triplets[k].valid = 1;
        }
        else
        {
            triplets[k].truth = triplet2prob[triplets[k]];
            triplets[k].valid = 0;
        }
    }
    
    for (int k = 0; k != entity_size; k++) h2rt[k].clear();
    
    int h, r, t;
    Pair ent_rel_pair;
    for (int k = 0; k != triplet_size; k++)
    {
        triplet2id[triplets[k]] = k;
        
        if (triplets[k].valid == 0) continue;
        
        h = triplets[k].h; r = triplets[k].r; t = triplets[k].t;
        
        ent_rel_pair.e = t;
        ent_rel_pair.r = r;
        h2rt[h].push_back(ent_rel_pair);
    }
}

void link_composition_rule(int id)
{
    int tid, h, mid, t;
    int rp0, rp1, rh;
    Triplet triplet;
    
    rp0 = rules[id].r_premise[0];
    rp1 = rules[id].r_premise[1];
    rh = rules[id].r_hypothesis;
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].valid == 0) continue;
        if (triplets[k].r != rp0) continue;
        
        h = triplets[k].h;
        mid = triplets[k].t;
        
        for (int i = 0; i != int(h2rt[mid].size()); i++)
        {
            if (h2rt[mid][i].r != rp1) continue;
            
            t = h2rt[mid][i].e;
            triplet.h = h; triplet.r = rh; triplet.t = t;
            
            if (triplet2id.count(triplet) == 0) continue;
            tid = triplet2id[triplet];
            sem_wait(&mutex);
            triplets[tid].rule_ids.push_back(id);
            sem_post(&mutex);
        }
    }
}

void link_symmetric_rule(int id)
{
    int tid, h, t, rp, rh;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].valid == 0) continue;
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (triplet2id.count(triplet) == 0) continue;
        tid = triplet2id[triplet];
        sem_wait(&mutex);
        triplets[tid].rule_ids.push_back(id);
        sem_post(&mutex);
    }
}

void link_inverse_rule(int id)
{
    int tid, h, t, rp, rh;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].valid == 0) continue;
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (triplet2id.count(triplet) == 0) continue;
        tid = triplet2id[triplet];
        sem_wait(&mutex);
        triplets[tid].rule_ids.push_back(id);
        sem_post(&mutex);
    }
}

void link_subrelation_rule(int id)
{
    int tid, h, t, rp, rh;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].valid == 0) continue;
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = h; triplet.t = t; triplet.r = rh;
        
        if (triplet2id.count(triplet) == 0) continue;
        tid = triplet2id[triplet];
        sem_wait(&mutex);
        triplets[tid].rule_ids.push_back(id);
        sem_post(&mutex);
    }
}

void *link_rules_thread(void *id)
{
    int thread = int((long)(id));
    int bg = int(rule_size / num_threads) * thread;
    int ed = int(rule_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = rule_size;
    
    for (int k = bg; k != ed; k++)
    {
        if (k % 10 == 0)
        {
            total_count += 10;
            printf("Progress: %.3lf%%          %c", (double)total_count / (double)(rule_size + 1) * 100, 13);
            fflush(stdout);
        }

        if (rules[k].type == "composition") link_composition_rule(k);
        if (rules[k].type == "symmetric") link_symmetric_rule(k);
        if (rules[k].type == "inverse") link_inverse_rule(k);
        if (rules[k].type == "subrelation") link_subrelation_rule(k);
    }
    
    pthread_exit(NULL);
}

void link_rules()
{
    sem_init(&mutex, 0, 1);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    total_count = 0;
    for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, link_rules_thread, (void *)a);
    for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    free(pt);

    printf("Data preprocessing done!          \n");
}

void init_weight()
{
    for (int k = 0; k != rule_size; k++)
    rules[k].weight = (rand() / double(RAND_MAX) - 0.5) / 100;
}

double train_epoch(double lr)
{
    double error = 0, cn = 0;
    
    for (int k = 0; k != rule_size; k++) rules[k].grad = 0;
    
    for (int k = 0; k != triplet_size; k++)
    {
        int len = int(triplets[k].rule_ids.size());
        if (len == 0) continue;
        
        triplets[k].logit = 0;
        for (int i = 0; i != len; i++)
        {
            int rule_id = triplets[k].rule_ids[i];
            triplets[k].logit += rules[rule_id].weight / len;
        }
        
        triplets[k].logit = sigmoid(triplets[k].logit);
        for (int i = 0; i != len; i++)
        {
            int rule_id = triplets[k].rule_ids[i];
            rules[rule_id].grad += (triplets[k].truth - triplets[k].logit) / len;
        }
        
        error += (triplets[k].truth - triplets[k].logit) * (triplets[k].truth - triplets[k].logit);
        cn += 1;
    }
    
    for (int k = 0; k != rule_size; k++) rules[k].weight += lr * rules[k].grad;
    
    return sqrt(error / cn);
}

void output_rules()
{
    if (output_rule_file[0] == 0) return;
    
    FILE *fo = fopen(output_rule_file, "wb");
    for (int k = 0; k != rule_size; k++)
    {
        std::string type = rules[k].type;
        double weight = rules[k].weight;
        
        fprintf(fo, "%s\t%s\t", type.c_str(), id2rel[rules[k].r_hypothesis].c_str());
        for (int i = 0; i != int(rules[k].r_premise.size()); i++)
        fprintf(fo, "%s\t", id2rel[rules[k].r_premise[i]].c_str());
        fprintf(fo, "%lf\n", weight);
    }
    fclose(fo);
}

void output_predictions()
{
    if (output_prediction_file[0] == 0) return;
    
    FILE *fo = fopen(output_prediction_file, "wb");
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].type == 'o') continue;
        
        int h = triplets[k].h;
        int t = triplets[k].t;
        int r = triplets[k].r;
        double prob = triplets[k].logit;
        
        fprintf(fo, "%s\t%s\t%s\t%lf\n", id2ent[h].c_str(), id2rel[r].c_str(), id2ent[t].c_str(), prob);
    }
    fclose(fo);
}

void output_hidden_triplets()
{
    if (output_hidden_file[0] == 0) return;
    
    FILE *fo = fopen(output_hidden_file, "wb");
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].type == 'o') continue;
        
        int h = triplets[k].h;
        int t = triplets[k].t;
        int r = triplets[k].r;
        
        fprintf(fo, "%s\t%s\t%s\n", id2ent[h].c_str(), id2rel[r].c_str(), id2ent[t].c_str());
    }
    fclose(fo);
}

void save()
{
    if (save_file[0] == 0) return;
    
    FILE *fo = fopen(save_file, "wb");
    
    fprintf(fo, "%d\n", entity_size);
    for (int k = 0; k != entity_size; k++) fprintf(fo, "%d\t%s\n", k, id2ent[k].c_str());
    
    fprintf(fo, "%d\n", relation_size);
    for (int k = 0; k != relation_size; k++) fprintf(fo, "%d\t%s\n", k, id2rel[k].c_str());
    
    fprintf(fo, "%d\n", triplet_size);
    for (int k = 0; k != triplet_size; k++)
    {
        int h = triplets[k].h;
        int r = triplets[k].r;
        int t = triplets[k].t;
        char type = triplets[k].type;
        int valid = triplets[k].valid;
        
        fprintf(fo, "%s\t%s\t%s\t%c\t%d\n", id2ent[h].c_str(), id2rel[r].c_str(), id2ent[t].c_str(), type, valid);
    }
    
    fprintf(fo, "%d\n", rule_size);
    for (int k = 0; k != rule_size; k++)
    {
        std::string type = rules[k].type;
        double weight = rules[k].weight;
        
        fprintf(fo, "%s\t%lf\t%s\t%d\t", type.c_str(), rules[k].precision, id2rel[rules[k].r_hypothesis].c_str(), int(rules[k].r_premise.size()));
        for (int i = 0; i != int(rules[k].r_premise.size()); i++)
        fprintf(fo, "%s\t", id2rel[rules[k].r_premise[i]].c_str());
        fprintf(fo, "%lf\n", weight);
    }
    
    fclose(fo);
}

void load()
{
    if (load_file[0] == 0) return;
    
    FILE *fi = fopen(load_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: loading file not found!\n");
        exit(1);
    }
    
    fscanf(fi, "%d", &entity_size);
    id2ent.clear(); ent2id.clear();
    int eid; char s_ent[MAX_STRING];
    for (int k = 0; k != entity_size; k++)
    {
        fscanf(fi, "%d %s", &eid, s_ent);
        id2ent.push_back(s_ent);
        ent2id[s_ent] = eid;
    }
    
    fscanf(fi, "%d", &relation_size);
    id2rel.clear(); rel2id.clear();
    int rid; char s_rel[MAX_STRING];
    for (int k = 0; k != relation_size; k++)
    {
        fscanf(fi, "%d %s", &rid, s_rel);
        id2rel.push_back(s_rel);
        rel2id[s_rel] = rid;
    }
    
    fscanf(fi, "%d", &triplet_size);
    triplets.clear();
    observed_triplets.clear();
    h2rt = new std::vector<Pair> [entity_size];
    int h, r, t;
    char t_type, s_head[MAX_STRING], s_tail[MAX_STRING];
    int valid;
    Triplet triplet;
    Pair ent_rel_pair;
    observed_triplet_size = 0; hidden_triplet_size = 0;
    for (int k = 0; k != triplet_size; k++)
    {
        fscanf(fi, "%s %s %s %c %d\n", s_head, s_rel, s_tail, &t_type, &valid);
        h = ent2id[s_head]; r = rel2id[s_rel]; t = ent2id[s_tail];
        triplet.h = h; triplet.r = r; triplet.t = t; triplet.type = t_type; triplet.valid = valid;
        triplet.rule_ids.clear();
        triplets.push_back(triplet);
        
        if (t_type == 'o')
        {
            observed_triplets.insert(triplet);
            observed_triplet_size += 1;
        }
        else
        {
            hidden_triplet_size += 1;
        }
        
        if (valid == 0) continue;
        ent_rel_pair.e = t;
        ent_rel_pair.r = r;
        h2rt[h].push_back(ent_rel_pair);
    }
    
    fscanf(fi, "%d", &rule_size);
    rules.clear();
    Rule rule;
    char r_type[MAX_STRING];
    for (int k = 0; k != rule_size; k++)
    {
        int cn;
        fscanf(fi, "%s %lf %s %d", r_type, &rule.precision, s_rel, &cn);
        rule.r_hypothesis = rel2id[s_rel];
        rule.type = r_type;
        rule.r_premise.clear();
        for (int i = 0; i != cn; i++)
        {
            fscanf(fi, "%s", s_rel);
            rule.r_premise.push_back(rel2id[s_rel]);
        }
        fscanf(fi, "%lf", &rule.weight);
        rules.push_back(rule);
    }
    
    fclose(fi);
    
    printf("#Entities: %d          \n", entity_size);
    printf("#Relations: %d          \n", relation_size);
    printf("#Observed triplets: %d          \n", observed_triplet_size);
    printf("#Hidden triplets: %d          \n", hidden_triplet_size);
    printf("#Triplets: %d          n", triplet_size);
    printf("#Rules: %d          \n", rule_size);
}

void train()
{
    if (load_file[0] == 0)
    {
        // Read observed triplets
        read_data();
        // Search for candidate logic rules
        search_candidate_rules();
        // Compute the empirical precision of logic rules and filter out low-precision ones
        compute_rule_precision();
        // Search for hidden triplets with the extracted logic rules
        search_hidden_triplets();
    }
    else
    {
        load();
    }
    
    save();
    output_hidden_triplets();

    if (iterations == 0) return;

    // Read the probability of hidden triplets predicted by KGE models
    read_probability_of_hidden_triplets();
    // Link each triplet to logic rules which can extract the triplet
    link_rules();
    // Initialize the weight of logic rules randomly
    init_weight();
    for (int k = 0; k != iterations; k++)
    {
        double error = train_epoch(learning_rate);
        printf("Iteration: %d %lf          \n", k, error);
    }
    output_rules();
    output_predictions();
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) 
    {
        if (a == argc - 1) 
        {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv)
{
    int i;
    if (argc == 1)
    {
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-observed <file>\n");
        printf("\t\tFile of observed triplets, one triplet per line, with the format: <h> <r> <t>.\n");
        printf("\t-probability <file>\n");
        printf("\t\tAnnotation of hidden triplets from KGE model, one triplet per line, with the format: <h> <r> <t> <prob>.\n");
        printf("\t-out-rule <file>\n");
        printf("\t\tOutput file of logic rules.\n");
        printf("\t-out-prediction <file>\n");
        printf("\t\tOutput file of predictions on hidden triplets by MLN.\n");
        printf("\t-out-hidden <file>\n");
        printf("\t\tOutput file of discovered hidden triplets.\n");
        printf("\t-save <file>\n");
        printf("\t\tSaving file.\n");
        printf("\t-load <file>\n");
        printf("\t\tLoading file.\n");
        printf("\t-iterations <int>\n");
        printf("\t\tNumber of iterations for training.\n");
        printf("\t-lr <float>\n");
        printf("\t\tLearning rate.\n");
        printf("\t-thresh-rule <float>\n");
        printf("\t\tThreshold for logic rules. Logic rules whose empirical precision is less than the threshold will be filtered out.\n");
        printf("\t-thresh-triplet <float>\n");
        printf("\t\tThreshold for triplets. Hidden triplets whose probability is less than the threshold will be viewed as false ones.\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of running threads.\n");
        return 0;
    }
    observed_triplet_file[0] = 0;
    probability_file [0] = 0;
    output_rule_file[0] = 0;
    output_prediction_file[0] = 0;
    output_hidden_file[0] = 0;
    save_file[0] = 0;
    load_file[0] = 0;
    if ((i = ArgPos((char *)"-observed", argc, argv)) > 0) strcpy(observed_triplet_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-probability", argc, argv)) > 0) strcpy(probability_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out-rule", argc, argv)) > 0) strcpy(output_rule_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out-prediction", argc, argv)) > 0) strcpy(output_prediction_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out-hidden", argc, argv)) > 0) strcpy(output_hidden_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save", argc, argv)) > 0) strcpy(save_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-load", argc, argv)) > 0) strcpy(load_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-iterations", argc, argv)) > 0) iterations = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) learning_rate = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-thresh-rule", argc, argv)) > 0) rule_threshold = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-thresh-triplet", argc, argv)) > 0) triplet_threshold = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    train();
    return 0;
}
