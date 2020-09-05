import os
import sys
import time
import tqdm
import torch
import shutil
import argparse
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from explainer import explain

# GPU = 4
# GPU = 6
GPU = 7
# GPU = 8

LOAD = 3

# ITERATIONS = 2
# ITERATIONS = 5
# ITERATIONS = 6
ITERATIONS = 10

# RANGE = [0.00,0.20,0.40,0.60,0.80,1.00]
RANGE = [0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]
# RANGE = [0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.50,4.00,4.50,5.00,6.00]

def AUROC(predictions):
    data = sorted(predictions, key=lambda x:x[0], reverse=True)
    pred = [float(j[0]) for j in data]; label = [int(j[1]) for j in data]
    P = sum(label); N = len(label)-sum(label); TP, FP = 0, 0; curve = [(0.0, 0.0)]
    for i in range(len(data)):
        if label[i]:
            TP += 1
        else:
            FP += 1
        curve.append((float(FP)/N if N!=0 else 1.0,float(TP)/P if P!=0 else 1.0))
    curve = sorted(curve, key=lambda x:(x[0],-x[1])); auroc = 0.
    for i in range(1,len(curve)):
        auroc += (curve[i][0]-curve[i-1][0]) * (curve[i][1]+curve[i-1][1]) / 2
    return auroc if predictions else 0.0

def TOPACC(predictions):
    data = sorted(predictions, key=lambda x:x[0], reverse=True)
    pred = [float(j[0]) for j in data]; label = [int(j[1]) for j in data]
    P = sum(label); N = len(label)-sum(label); TP, FP = 0, 0; curve = [(0.0, 0.0)]; acc,pos = 0,-1
    for i in range(len(data)):
        if label[i]:
            TP += 1
        else:
            FP += 1
        if acc < (TP+N-FP)/(P+N):
            acc = (TP+N-FP)/(P+N)
            pos = i
    return (acc,pred[pos]) if pos>=0 else (0.0,0.0)

def THRESACC(predictions, threshold):
    return sum([int(p[0]>threshold)==int(p[1]) for p in predictions])/len(predictions) if predictions else 0.0

def DrawCurve(title, x, y, std=None):
    plt.figure(figsize=(16,9),dpi=300); plt.errorbar(x, y, yerr=std, fmt='bo-', elinewidth=2, capsize=3, ms = 5)
    plt.errorbar(x, np.array(y)-np.array(std), fmt='b:'); plt.errorbar(x, np.array(y)+np.array(std), fmt='b:')
    plt.xlabel("p"); plt.ylabel(title); plt.grid(axis='y'); plt.savefig("./report/{0}.png".format(title)); plt.close()
    # plt.figure(); plt.plot(x,y); plt.xlabel("p"); plt.ylabel(title); plt.savefig("./workspace/{0}.png".format(title)); plt.close()

def Preprocess(syn_type, p=0.0, h=0, s=998244353, feat_gen="Binomial", graph_only=False):
    return subprocess.Popen(('CUDA_VISIBLE_DEVICES=\'{:d}\' python3 train.py --name-suffix=\'p={:.2f}\''
                           +' --p={:.2f} --seed={:d} {:s} --feat-gen=\'{:s}\' --num-gc-layers={:d} --syn-type={:s}')
                           .format(h,p,p,s,"--graph-only" if graph_only else "",feat_gen,
                           4 if syn_type.split('-')[1]=='grid' else 3,syn_type),shell=True)

def ExplainInstance(syn_type, p, nodes, h=0):
    feat_nodes = torch.load("./workspace/feat_nodes_p={:.2f}.pkl".format(p))
    torch.save([node for node in nodes if not feat_nodes[node]],"./workspace/nodes-p={:.2f}-f=0.tmp".format(p))
    torch.save([node for node in nodes if feat_nodes[node]],"./workspace/nodes-p={:.2f}-f=1.tmp".format(p))
    return subprocess.Popen(('CUDA_VISIBLE_DEVICES=\'{:d}\' python3 explainer_main.py --name-suffix=\'p={:.2f}\''
                           +' --stat --cuda=\'{:d}\'  --num-gc-layers={:d} --syn-type={:s}').format(h,p,h,4 if syn_type.split('-')[1]=='grid' else 3,syn_type),shell=True)

def GetPredictions(p):
    stat = torch.load("./workspace/GNNE-p={:.2f}-f=0.stat".format(p)); predictions0 = [(p,y) for p,y in zip(stat['data'][0],stat['data'][1])]
    stat = torch.load("./workspace/GNNE-p={:.2f}-f=1.stat".format(p)); predictions1 = [(p,y) for p,y in zip(stat['data'][0],stat['data'][1])]
    # print(p,stat['AUROC'])
    return [predictions0,predictions1,predictions0+predictions1]
    
def StatisticsGNNE(predictions, per=0.35):
    TOPTHRES = [TOPACC(prediction[:int(len(prediction)*per)])[1] for prediction in predictions]
    result = [{'AUROC':AUROC(prediction),'0.7ACC':THRESACC(prediction,0.7),'0.8ACC':THRESACC(prediction,0.8),'0.9ACC':THRESACC(prediction,0.9),
               'TOPTHRES':TOPTHRES[i],'TOPACC':THRESACC(prediction[int(len(prediction)*per):],TOPTHRES[i])} for i,prediction in enumerate(predictions)]
    mean_result = {key:float(np.mean([r[key] for r in result]) if result else 0.0) for key in ['AUROC','TOPACC','TOPTHRES','0.7ACC','0.8ACC','0.9ACC']}
    std_result  = {key:float(np.std([r[key] for r in result]) if result else 0.0) for key in ['AUROC','TOPACC','TOPTHRES','0.7ACC','0.8ACC','0.9ACC']}
    return mean_result, std_result

def StatisticsGNN(predictions):
    result = [{'ACC':(TP+TN)/(TP+FP+TN+FN) if (TP+FP+TN+FN) else 0.0,'PRE':TP/(TP+FP) if (TP+FP) else 0.0,'REC':TP/(TP+FN) if (TP+FN) else 0.0}
               for TP,FP,TN,FN in predictions]
    mean_result = {key:float(np.mean([r[key] for r in result]) if result else 0.0) for key in ['ACC','PRE','REC']}
    std_result  = {key:float(np.std([r[key] for r in result]) if result else 0.0) for key in ['ACC','PRE','REC']}
    return mean_result, std_result

def Report(R, args):
    GNNR = [[],[],[]]; GNN = [[{key:[] for key in ['ACC','PRE','REC']} for _ in range(2)] for _ in range(3)]
    GNNER = [[],[],[]]; GNNE = [[{key:[] for key in ['AUROC','TOPACC','TOPTHRES','0.7ACC','0.8ACC','0.9ACC']} for _ in range(2)] for _ in range(3)]
    for h,p in enumerate(R,1):
        for i in [0,1,2]:
            if (i==0 and p==1.00) or (i==1 and p==0.00):
                pass
            elif os.path.exists("./workspace/GNN-p={:.2f}-f={:d}-performance.pkl".format(p,i)):
                result = torch.load("./workspace/GNN-p={:.2f}-f={:d}-performance.pkl".format(p,i))
                for key in ['ACC','PRE','REC']:
                    GNN[i][0][key].append(result[0][key])
                    GNN[i][1][key].append(result[1][key])
                GNNR[i].append(p)
        if not args.gnn_only:
            for i in [0,1,2]:
                if (i==0 and p==1.00) or (i==1 and p==0.00):
                    pass
                elif os.path.exists("./workspace/GNNE-p={:.2f}-f={:d}-performance.pkl".format(p,i)):
                    result = torch.load("./workspace/GNNE-p={:.2f}-f={:d}-performance.pkl".format(p,i))
                    for key in ['AUROC','TOPACC','TOPTHRES','0.7ACC','0.8ACC','0.9ACC']:
                        GNNE[i][0][key].append(result[0][key])
                        GNNE[i][1][key].append(result[1][key])
                    GNNER[i].append(p)
    for i in [0,1,2]:
        for key in ['ACC','PRE','REC']:
            None if i==1 and key=='PRE' else DrawCurve("GNN-{1}-f={0}".format(i,key),GNNR[i],GNN[i][0][key],GNN[i][1][key])
        if not args.gnn_only:
            for key in ['AUROC','TOPACC','TOPTHRES','0.7ACC','0.8ACC','0.9ACC']:
                DrawCurve("GNNE-{1}-f={0}".format(i,key),GNNER[i],GNNE[i][0][key],GNNE[i][1][key])

def RunInstances(R, args):
    start_time = time.time()
    print("Start:",time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(start_time)))
    shutil.rmtree("./workspace/") if os.path.exists("./workspace/") else None; os.mkdir("./workspace/")
    shutil.rmtree("./log/") if os.path.exists("./log/") else None; os.mkdir("./log/")
    shutil.rmtree("./ckpt/") if os.path.exists("./ckpt/") else None; os.mkdir("./ckpt/")
    handles = []; preds = [[[], [], []] for _ in range(len(R)+1)]; M = [[[], [], []] for _ in range(len(R)+1)]; Preprocess(args.syn_type,0,0,0,graph_only=True).wait()
    ### Operation Inserted Here
    
    ###
    for iteration in range(ITERATIONS):
        for h,p in enumerate(R,1):
            handles.append(Preprocess(args.syn_type,p=p,h=h%GPU,s=np.random.randint(1000000000),feat_gen=args.feat_gen))
            if h%(LOAD*GPU)==0 or h==len(R):
                _, handles = [handle.wait() for handle in handles], []
        for h,p in enumerate(R,1):
            result = [torch.load("./workspace/GNN-p={:.2f}-f={:d}-performance.tmp".format(p,i)) for i in range(3)]
            for i in range(3):
                M[h][i].append(result[i])
                torch.save(StatisticsGNN(M[h][i]),"./workspace/GNN-p={:.2f}-f={:d}-performance.pkl".format(p,i))

        if not args.gnn_only:
            for h,p in enumerate(R,1):
                handles.append(ExplainInstance(args.syn_type,p,[v for v in range(1023,1023+720)],h%GPU))
                if h%(LOAD*GPU)==0 or h==len(R):
                    _, handles = [handle.wait() for handle in handles], []
            for h,p in enumerate(R,1):
                result = GetPredictions(p)
                for i in range(3):
                    preds[h][i].append(result[i])
                    torch.save(StatisticsGNNE(preds[h][i]),"./workspace/GNNE-p={:.2f}-f={:d}-performance.pkl".format(p,i))
        
        Report(R,args)
        print("Iteration {0}/{1} Finished:".format(iteration+1,ITERATIONS),time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(time.time())))

    os.remove("./workspace/graph.pkl")
    os.remove("./workspace/struct_nodes.pkl")
    for h,p in enumerate(R,1):
        os.remove("./workspace/feat_nodes_p={:.2f}.pkl".format(p))
        os.remove("./workspace/labeled_nodes_p={:.2f}.pkl".format(p))
        for i in range(3):
            os.remove("./workspace/GNN-p={:.2f}-f={:d}-performance.tmp".format(p,i))
    end_time = time.time(); interval = int(end_time-start_time)
    print("Start:",time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(start_time)))
    print("End:",time.strftime("%Y-%m-%d %H:%M:%S ",time.localtime(end_time)))
    print("Total Time: {:02d}h{:02d}m{:02d}s".format(interval//3600,interval%3600//60,interval%60))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', dest='report_only', action='store_const', const=True, default=False,
                        help='If statistics already exists, use \'--report-only\' to generate report based on existing data instead of re-running.')
    parser.add_argument('--gnn-only', dest='gnn_only', action='store_const', const=True, default=False,
                        help='Use \'--gnn-only\; to run GNN but not GNNE.')
    parser.add_argument('--syn-type', dest='syn_type', default='tree-cycle', type=str,
                        help="Options:'tree-cycle', 'tree-grid', 'tree-house', 'ba-cycle', 'ba-grid', 'ba-house'")
    parser.add_argument('--feat-gen', dest='feat_gen', default='Binomial', type=str,
                        help="Options:'tree-cycle', 'tree-grid', 'tree-house', 'ba-cycle', 'ba-grid', 'ba-grid'")
    args = parser.parse_args()
    shutil.rmtree("./report/") if os.path.exists("./report/") else None; os.mkdir("./report/")
    if args.report_only:
        Report(RANGE,args)
    else:
        RunInstances(RANGE,args)