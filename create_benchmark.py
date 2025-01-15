#!/usr/bin/env python
####!/usr/bin/python
'''
Program: creating CAFA benchmark
Author : Huy Nguyen
Start  : 05/31/2017
End    : 05/31/2017
'''
import os
import sys
import argparse
import json
import obonet
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
from scipy.sparse import dok_matrix
from Bio.UniProt import GOA
from Bio import SwissProt as sp

# evidence from experimental
with open('configGOcodes.json', 'r') as config_file:
        config = json.load(config_file)
selected_codes = {'Evidence': set(config['Experimental'])}

def parse_inputs(argv):
    parser = argparse.ArgumentParser(
        description='Compare difference between 2 GOA files or 2 .dat files. If annotations in input file have been propagated to ontology roots, the input onotology graph should be the same as the one used to propagate terms')
    
    parser.add_argument('--annot1', '-a1', required=True,
                        help='Path to first annotation file')
    parser.add_argument('--annot2', '-a2', required=True,
                        help='Path to second annotation file')
    parser.add_argument('--filetype', '-t', required=True, choices=['goa', 'dat'], 
                        help='Input file type')
    parser.add_argument('--graph', '-g', default=None, 
                        help='Path to OBO ontology graph file if local. If empty (default) current OBO structure at run-time will be downloaded from http://purl.obolibrary.org/obo/go/go-basic.obo')
    parser.add_argument('--outfile', '-o', default='diff.txt', 
                        help='Path to save computed IA for each term in the GO. If empty, will be saved to ./IA.txt')  
    return parser.parse_args(argv)


def process_gaf_file(gaf_file):
    '''
    function : given a file handle, parse in using gaf format and return a dictionary
            that identify those protein with experimental evidence and the ontology
    input    : file text
    output   : dic (key: name of file (number), value is a big dictionary store info about the protein)
    '''
    with open(gaf_file, 'r') as f:
        content = f.read()
    # Find the position of "!gaf-version" line
    gaf_version_pos = content.find("!gaf-version")
    if gaf_version_pos == -1:
        # If "!gaf-version" is not found, process from the beginning
        return content
    else:
        # If "!gaf-version" is found, process from that line onwards
        return content[gaf_version_pos:]


def clean_ontology_edges(ontology):
    """
    Remove all ontology edges except types "is_a" and "part_of" and ensure there are no inter-ontology edges
    :param ontology: Ontology stucture (networkx DiGraph or MultiDiGraph)
    """
    
    # keep only "is_a" and "part_of" edges (All the "regulates" edges are in BPO)
    remove_edges = [(i, j, k) for i, j, k in ontology.edges if not(k=="is_a" or k=="part_of")]
    
    ontology.remove_edges_from(remove_edges)
    
    # There should not be any cross-ontology edges, but we verify here
    crossont_edges = [(i, j, k) for i, j, k in ontology.edges if
                      ontology.nodes[i]['namespace']!= ontology.nodes[j]['namespace']]
    if len(crossont_edges)>0:
        ontology.remove_edges_from(crossont_edges)
    
    return ontology
    

def fetch_aspect(ontology, root:str):
    """
    Return a subgraph of an ontology starting at node <root>
    
    :param ontology: Ontology stucture (networkx DiGraph or MultiDiGraph)
    :param root: node name (GO term) to start subgraph
    """
    namespace = ontology.nodes[root]['namespace']
    aspect_nodes = [n for n,v in ontology.nodes(data=True) 
                    if v['namespace']==namespace]
    subont_ = ontology.subgraph(aspect_nodes)
    return subont_


def propagate_terms(terms_df, subontologies):
    """
    Propagate terms in DataFrame terms_df abbording to the structure in subontologies.
    If terms were already propagated with the same graph, the returned dataframe will be equivalent to the input
    
    :param terms_df: pandas DataFrame of annotated terms (column names 'EntryID', 'term' 'aspect')
    :param subontologies: dict of ontology aspects (networkx DiGraphs or MultiDiGraphs)
    """
    # Look up ancestors ahead of time for efficiency
    subont_terms = {aspect: set(terms_df[terms_df.aspect==aspect].term.values) for aspect in subontologies.keys()}
    ancestor_lookup = {aspect:{t: nx.descendants(subont,t) for t in subont_terms[aspect]
                             if t in subont} for aspect, subont in subontologies.items()}

    propagated_terms = []
    for (protein, aspect), entry_df in terms_df.groupby(['EntryID', 'aspect']):
        protein_terms = set().union(*[list(ancestor_lookup[aspect][t])+[t] for t in set(entry_df.term.values)])

        propagated_terms += [{'EntryID': protein, 'term': t, 'aspect': aspect} for t in protein_terms]

    return pd.DataFrame(propagated_terms)


def term_counts(terms_df, term_indices):
    """
    Count the number of instances of each term
    
    :param terms_df: pandas DataFrame of (propagated) annotated terms (column names 'EntryID', 'term', 'aspect')
    :param term_indices: dict mapping term names to column indices
    """
    num_proteins = len(terms_df.groupby('EntryID'))
    S = dok_matrix((num_proteins+1, len(term_indices)), dtype=np.int32)
    S[-1,:] = 1  # dummy protein
    
    for i, (protein, protdf) in enumerate(terms_df.groupby('EntryID')):
        row_count = {term_indices[t]:c for t,c in Counter(protdf['term']).items()}
        for col, count in row_count.items():
            S[i, col] = count
    
    return S


def read_gaf(handle):
    # Add preprocessing for GAF files
    gaf_content = process_gaf_file(handle)
    temp_gaf_file = handle + '.temp'
    with open(temp_gaf_file, 'w') as f:
        f.write(gaf_content)
    name = handle.split(".")[-1]
    
    all_protein_name = set()
    # Experimental_codes = {'Evidence': set(['EXP','IDA','IPI','IMP','IGI','IEP'])}
    data = []
    with open(temp_gaf_file, 'r') as handle:
        for rec in GOA.gafiterator(handle):
            # Remove NOT annotations
            if 'NOT' in rec['Qualifier']:
                continue
            all_protein_name.add(rec['DB_Object_ID'])
            # Add the ancestral terms to the dictionary in t1
            if GOA.record_has(rec, selected_codes) and rec['DB'] == 'UniProtKB':
                data.append({'EntryID': rec['DB_Object_ID'], 'term': rec['GO_ID'], 'aspect': rec['Aspect'], 'evidence': rec['Evidence']})
    
    os.remove(temp_gaf_file)
    df = pd.DataFrame(data)
    return name, df, all_protein_name


def process_go_from_dat(file_path):
    entries = []
    with open(file_path, 'r') as file:
        for record in sp.parse(file):
            current_id = record.accessions[0]
            for dr in record.cross_references:    #dr -> db cross refernce
                if dr[0] == 'GO' and len(dr) >= 4:
                    go_id = dr[1]
                    aspect = dr[2][0]  # Getting only the first letter (its either P/ C/ F)
                    # aspect_description = dr[2][2:]  # Getting the rest of the description
                    evidence = dr[3] if len(dr) >= 4 else ''
                    evidence_code = evidence[:3]  # First 3 letters of evidence
                    if evidence_code not in selected_codes['Evidence']:
                        continue
                    # evidence_source = evidence[4:] if len(evidence) > 4 else ''
                    entries.append({
                        "EntryID": current_id,
                        "term": go_id,
                        "aspect": aspect,
                        #"Aspect_Description": aspect_description.strip(),
                        "evidence": evidence_code.strip(),
                        #"Evidence Source": evidence_source.strip()
                    })
    # with open(output_path, 'w', newline='') as tsvfile:
    #     #fieldnames = ["ID", "GO ID", "Aspect", "Aspect_Description", "Evidence Code", "Evidence Source"]
    #     fieldnames = ["ID", "GO ID", "Aspect", "Evidence Code"]
    #     writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
    #     writer.writeheader()
    #     writer.writerows(entries)
    df = pd.DataFrame(entries)
    return df


def compare_leaves(df1, df2, output_file):
    # df_2024 = pd.read_csv(file_2024, sep="\t")
    # df1 = pd.read_csv(file_2019, sep="\t")

    df2 = df2[['EntryID', 'term', 'aspect', 'evidence']]
    df1 = df1[['EntryID', 'term', 'aspect', 'evidence']]
    
    # Groupingg by ID and GO ID for comparison
    grouped_2 = df2.groupby(['EntryID', 'term']).first().reset_index()
    grouped_1 = df1.groupby(['EntryID', 'term']).first().reset_index()

    # Mergingg th e datasets on ID and GO ID
    merged = pd.merge(
        grouped_2,
        grouped_1,
        on=['EntryID', 'term'],
        how='outer',
        suffixes=('_t2', '_t1'),
        indicator=True
    )

    # Determininggg the type of change for each entryyy
    def detect_changes(row):
        if row['_merge'] == 'left_only':
            return 'Added' #if a new GO ID is found which didn't exist earlierr, its represented byyy Added
        elif row['_merge'] == 'right_only':
            return 'Removed' #if there was a GO ID earlier, and now discarded from the new db, its representeddd by Removed
        else:
            if (row['aspect_t2'] != row['aspect_t1'] or
                row['evidence_t2'] != row['evidence_t1']):
                return 'Modified' #if the GO ID remains the same, but there's an update/modificationn in the Evidence Code
            return 'Unchanged' #if the GOID remains the same/unchanged from both the time periodss

    merged['change_type'] = merged.apply(detect_changes, axis=1)

    result = merged[['EntryID', 'term', 'aspect_t2', 'aspect_t1',
                     'evidence_t2', 'evidence_t1', 'change_type']]
    # result.rename(
    #     columns={
    #         'Aspect_2024': 'Aspect (2024)',
    #         'Aspect_2019': 'Aspect (2019)',
    #         'Evidence Code_2024': 'EC (2024)',
    #         'Evidence Code_2019': 'EC (2019)'
    #     },
    #     inplace=True
    # )
    result['aspect'] = np.where(result['change_type'] == 'Added', 
                                  result['aspect_t2'], 
                                  result['aspect_t1'])
    result.to_csv(output_file, sep="\t", index=False)
    # group by EntryID and aspect, check if any of 'change_type' is 'Added' or 'Removed' then keep, otherwise remove
    result_changed = result.copy()
    mask_changed = result_changed.groupby(['EntryID', 'aspect'])['change_type'].transform(lambda x: x.isin(['Added', 'Removed']).any())    
    result_ECmodified = result[result['change_type'] == 'Modified'].drop(columns=['aspect_t2', 'aspect_t1'])
    mask_unchanged = result.groupby(['EntryID', 'aspect'])['change_type'].transform(lambda x: x.isin(['Unchanged']).all())
    result_changed = result_changed.loc[mask_changed].drop(columns=['aspect_t2', 'aspect_t1', 'evidence_t2', 'evidence_t1'])
    result_unchanged = result.loc[mask_unchanged].drop(columns=['term','aspect_t2', 'aspect_t1', 'evidence_t2', 'evidence_t1'].drop_duplicates())
    result_unchanged.to_csv('test_unchanged.tsv', sep="\t", index=False)
    result_ECmodified.to_csv('test_ECmodified.tsv', sep="\t", index=False)
    print(f"Total proteins: {len(result['EntryID'].unique())}. Proteins with ancestral changes: {len(result_changed['EntryID'].unique())}")
    
    return result_changed
    
        
def analyze(t1_dic,t2_dic,all_protein_t1):
    '''
    function : given t1 dic, t2 dic, we provide the dic for NK, and LK dic for each ontology
    input    : 2 dics
    output   : NK,LK dictionary
    '''
    NK_dic = {'P':{},'C':{},'F':{}}
    LK_dic = {'P':{},'C':{},'F':{}}
    # dealing with NK and LK
    
    for protein in t2_dic:
        ## check the protein in t2_dic but not appear in t1
        if protein not in t1_dic and protein in all_protein_t1: ## this going to be in NK
            ### check which ontology got new annotated
            for ontology in t2_dic[protein]:
                NK_dic[ontology][protein] = t2_dic[protein][ontology]
        ## check the protein that in t2_dic and appear in t1
        elif protein  in t1_dic :
            ## check if in t1, this protein does not have all 3 ontology
            ### if yes, then not include since full knowledge
            ### else
            if len(t1_dic[protein]) < 3:
                #### check if t2_dic include in the ontology that t1 lack of
                for ontology in t2_dic[protein]:
                    if ontology not in t1_dic[protein]: # for those lack, include in LK
                        LK_dic[ontology][protein] = t2_dic[protein][ontology]
    # NK: no experimental terms in 3 aspects in t1, gain in t2 aspect-specific
    # LK: had experimental terms in 1 or 2 aspects in t1, gain in t2 aspect-specific
    # Does not consider ancestral terms, but proteins never gain old terms (if it has a term before, its not in NK or LK for that aspect)
    # Possible improvements: any non-IEA instead of EXPEC, gained more specific terms in t2 compared to t1 (full knowledge in that aspect)
    # Partial knowledge: it gained in the same aspect                    
    return NK_dic,LK_dic 
    
    
def write_file(dic,knowledge,name):
    '''
    function : given NK,LK dic , write out 6 files 
    input    : 2 dics
    output   : NK,LK dictionary
    '''
    for ontology in dic:
        if ontology =='F':
            final_name = name+knowledge+'_mfo'
        elif ontology =='P':
            final_name = name+knowledge+'_bpo'
        elif ontology =='C':
            final_name = name+knowledge+'_cco'
        print("Writing {} file".format(final_name))
        file_out = open(final_name,'w')
        for protein in sorted(dic[ontology]):
            for annotation in dic[ontology][protein]:
                file_out.write(protein +'\t'+annotation+'\n')
        file_out.close()
    return None
    

if __name__ == "__main__":
    args = parse_inputs(sys.argv[1:])
    # load ontology graph and GO terms
    ontology_graph = clean_ontology_edges(obonet.read_obo(args.graph))
    roots = {'P': 'GO:0008150', 'C': 'GO:0005575', 'F': 'GO:0003674'}
    subontologies = {aspect: fetch_aspect(ontology_graph, roots[aspect]) for aspect in roots}
    
    if args.filetype == 'goa':
        t1_name,annotation_df1,all_protein_t1 = read_gaf(args.annot1)
        t2_name,annotation_df2,all_protein_t2 = read_gaf(args.annot2)
        ancestral_change = compare_leaves(annotation_df1, annotation_df2, 'test_leaves.tsv')
    elif args.filetype == 'dat':
        annotation_df1 = process_go_from_dat(args.annot1)
        annotation_df2 = process_go_from_dat(args.annot2)
        ancestral_change = compare_leaves(annotation_df1, annotation_df2, 'test_leaves.tsv')
    
    # Guardrail for terms that didn't exist in t1
    all_terms_t2 = annotation_df2.groupby('aspect')['term'].unique().apply(set)
    new_terms = {aspect: all_terms_t2.get(aspect, set()) - set(subontologies[aspect].nodes()) for aspect in roots}
    all_new_terms = set.union(*new_terms.values())
    mask_new_terms = annotation_df2['term'].isin(all_new_terms)
    new_terms_df = {}
    for aspect in subontologies:
        aspect_mask = (annotation_df2['aspect'] == aspect) & mask_new_terms
        df = annotation_df2[aspect_mask].copy()
        df.loc[:, 'change_type'] = 'New'
        new_terms_df[aspect] = df
    # Filter out new terms for propagation
    annotation_df2 = annotation_df2[~mask_new_terms]
    # Only propagate terms for protein-aspect that has changes
    protein_changes = set(ancestral_change['EntryID'].unique())
    # Propagate leaf terms
    print('Propagating Terms')
    annotation_df1 = propagate_terms(annotation_df1[annotation_df1['EntryID'].isin(protein_changes)], subontologies)
    annotation_df2 = propagate_terms(annotation_df2[annotation_df2['EntryID'].isin(protein_changes)], subontologies)

    # Count term instances
    aspect_terms = dict()
    # Collect all unique terms for each aspect
    all_terms = {aspect: set() for aspect in subontologies}
    for aspect, subont in subontologies.items():
        all_terms[aspect].update(annotation_df1[annotation_df1.aspect==aspect]['term'].unique())
        all_terms[aspect].update(annotation_df2[annotation_df2.aspect==aspect]['term'].unique())
    
    # Create a unified term index for each aspect
    term_idx = {}
    for aspect in subontologies:
        aspect_terms = sorted(all_terms[aspect])
        term_idx[aspect] = {t:i for i,t in enumerate(aspect_terms)}

    # Collect all unique proteins
    all_proteins = set(annotation_df1['EntryID'].unique()).union(annotation_df2['EntryID'].unique())
    
    # Create a protein index
    protein_idx = {p:i for i,p in enumerate(sorted(all_proteins))}

    # Count term instances using the unified term index
    print('Counting Terms for annotation file 1')
    aspect_counts1 = dict()
    for aspect in subontologies:
        # Use the same protein index and term index for both matrices
        num_proteins = len(protein_idx)
        S = dok_matrix((num_proteins+1, len(term_idx[aspect])), dtype=np.int32)
        S[-1,:] = 1  # dummy protein
        for i, (protein, protdf) in enumerate(annotation_df1[annotation_df1.aspect==aspect].groupby('EntryID')):
            row_count = {term_idx[aspect][t]:c for t,c in Counter(protdf['term']).items()}
            for col, count in row_count.items():
                S[protein_idx[protein], col] = count
        aspect_counts1[aspect] = S
        
    print('Counting Terms for annotation file 2')
    aspect_counts2 = dict()
    for aspect in subontologies:
        # Use the same protein index and term index for both matrices
        num_proteins = len(protein_idx)
        S = dok_matrix((num_proteins+1, len(term_idx[aspect])), dtype=np.int32)
        S[-1,:] = 1  # dummy protein
        for i, (protein, protdf) in enumerate(annotation_df2[annotation_df2.aspect==aspect].groupby('EntryID')):
            row_count = {term_idx[aspect][t]:c for t,c in Counter(protdf['term']).items()}
            for col, count in row_count.items():
                S[protein_idx[protein], col] = count
        aspect_counts2[aspect] = S
    
    # Convert to Compressed Sparse Row format for efficient comparison
    sp_matrix1 = {aspect:dok.tocsr() for aspect, dok in aspect_counts1.items()}
    sp_matrix2 = {aspect:dok.tocsr() for aspect, dok in aspect_counts2.items()}

    # Compute the difference matrix
    print('Computing Difference Matrix')
    diff_matrix = {}
    for aspect in subontologies.keys():
        diff_matrix[aspect] = sp_matrix2[aspect] - sp_matrix1[aspect]
     
    # Create reverse mapping for protein indices
    protein_idx_reverse = {i:p for p,i in protein_idx.items()}
    
    for aspect, matrix in diff_matrix.items():
        rows, cols = matrix.nonzero()
        
        # Create DataFrame with all necessary information
        df = pd.DataFrame({
            'row_idx': rows,
            'col_idx': cols,
            'value': matrix.data
        })
        
        # Map indices back to protein IDs and terms
        local_term_idx = {i:t for t,i in term_idx[aspect].items()}
        df['EntryID'] = df['row_idx'].apply(lambda x: protein_idx_reverse[x])
        df['term'] = df['col_idx'].apply(lambda x: local_term_idx[x])
        df['change_type'] = df['value'].apply(lambda x: 'Added' if x>0 else 'Removed')
        df['aspect'] = aspect
        # Concatenate with new terms dataframe
        df = pd.concat([df, new_terms_df[aspect]])
        # Save the difference matrix to file with protein IDs and terms
        df[['EntryID', 'term', 'aspect', 'change_type']].to_csv(f'{args.outfile}_{aspect}.tsv', sep='\t', index=False)
    # num1 = t1.split('.')[-1]
    # num2 = t2.split('.')[-1]
    # name = outdir+'/'+'.'.join(t1.split('/')[-1].split('.')[:-1])+'.'+num2+'-'+num1+'_benchmark_'
    # write_file(NK_dic,'NK',name)
    # write_file(LK_dic,'LK',name)
