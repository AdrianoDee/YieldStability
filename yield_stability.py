import pandas as pd

import uproot

from matplotlib import pyplot as plt

import numpy as np

import os

import sys

import argparse

#from string import split

runfill = lambda x :  float(x.split(":")[0])
fillrun = lambda x :  float(x.split(":")[1])

parser = argparse.ArgumentParser()
parser.add_argument('--lumi',  type=str, default="./lumisections",
                    help="Lumi section table path.")
parser.add_argument('--data',  type=str, default="./data.root",
                    help="ROOT file to be loeaded name.")
parser.add_argument('--dir',   type=str, default=None,
                    help="Directory, in ROOT file, where the tree is stored. If none, leave it blank.")
parser.add_argument('--tree',  type=str, default="tree",
                    help="Tree to be loaded.")
parser.add_argument('--run',  type=str, default="run",
                    help="Run leaf/column name (default = \'run\'")
parser.add_argument('--keys',  nargs="+",type=str, default=None,
                    help="Run leaf/column name (default = \'run\'")
args = parser.parse_args()

DUPLICATES = args.keys
print(DUPLICATES)
RUN = args.run

print("> Loading BrilCalc Lumi Table")
lumi_path = args.lumi
runs = pd.read_csv(args.lumi,delimiter=",")
runs.head()
runs[RUN] = runs["run:fill"].apply(runfill)
runs["fill"] = runs["run:fill"].apply(fillrun)
runs = runs.drop(["run:fill"],axis=1)

print("> Loading ROOT file into Pandas table")
tree = uproot.open(args.data)
if args.dir is not None:
    tree = tree[args.dir][args.tree]
else:
    tree = tree[args.tree]

tree = tree.pandas.df([RUN])


plt.figure(figsize=(12,9))

min_run = tree[RUN].min()
max_run = tree[RUN].max()

n,b = np.histogram(tree[RUN].values.astype(float),bins=1000,range=(min_run-200,max_run+200))
bw = (b[1]-b[0])*0.5
plt.errorbar(b[:-1] + bw,n,c="red",yerr=np.sqrt(n));

ax = plt.gca()
ax.margins(x=0,tight=False)
plt.ylim(0,)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.grid(True,axis="y")
plt.xlabel("Run Number",fontsize=17,fontstyle="italic")
plt.ylabel("Entries$",fontsize=17,fontstyle="italic")
#legend = plt.legend(title="Eras",fontsize=18,loc=2)
#plt.setp(legend.get_title(),fontsize=18,fontweight="bold")
#plt.barh(xnumbers,new_mean[99])
plt.title("Run Counts",fontsize=18,fontweight="bold")

plt.savefig('run_count.png')


counts_df = tree[RUN].value_counts().astype(float)
counts_df = pd.DataFrame(np.array([counts_df.index.astype(int),counts_df.values]).transpose(1,0),columns=[RUN,"counts"])
counts_df = counts_df[counts_df[RUN]>1] 
counts_df.head()


runs = pd.merge(counts_df, runs, on=RUN)
runs["ratio"] = runs["counts"]*1000.0/runs["recorded"] # from mb to µb

xnames = runs[RUN]

plt.figure(figsize=(12,9))
ax = plt.gca()

plt.plot(xnames,runs["ratio"].values,"o",color="blue");
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
plt.xlabel("Run Number",fontsize=17,fontstyle="italic")
plt.ylabel("Entries / $(µb)^{-1}$",fontsize=17,fontstyle="italic")
plt.ylim(0,)    
#legend = plt.legend(title="",fontsize=18,loc=2)
#plt.setp(legend.get_title(),fontsize=18,fontweight="bold")
#plt.barh(xnumbers,new_mean[99])
plt.title("Candidates Count",fontsize=22,fontstyle="italic")

plt.savefig('cand_count.png')

