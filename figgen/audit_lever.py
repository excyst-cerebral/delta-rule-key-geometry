import data as D, numpy as np, itertools, math
def acc(reg,lam): return {s:D.mq_acc(D.MQ[(reg,lam,s)][1]) for s in D.mq_seeds(reg,lam)}
b=acc("none",0.0); d=acc("dec",1.0)
ss=sorted(set(b)&set(d)); diff=np.array([d[s]-b[s] for s in ss]); print("n",len(ss),"mean diff",diff.mean().round(4),"higher",(diff>0).sum())
rng=np.random.default_rng(0); bs=[rng.choice(diff,len(diff)).mean() for _ in range(20000)]; print("boot CI",np.percentile(bs,[2.5,97.5]).round(3))
# exact sign-flip p (two-sided)
sf=[abs(np.sum(diff*np.array(sg)))/len(diff) for sg in itertools.product([-1,1],repeat=len(diff))]; print("signflip p",np.mean(np.array(sf)>=abs(diff.mean())-1e-12).round(4))
# exact Wilcoxon
r=np.argsort(np.argsort(abs(diff)))+1; W=r[diff>0].sum(); allW=[sum(rr for rr,sg in zip(r,sgs) if sg) for sgs in itertools.product([0,1],repeat=len(r))]; mu=sum(r)/2
print("wilcoxon p", np.mean(np.abs(np.array(allW)-mu)>=abs(W-mu)-1e-12).round(4))
order=np.argsort(-diff); print("largest gains", diff[order][:4].round(3), "baseline seeds", [ss[i] for i in order[:4]], [round(b[ss[i]],2) for i in order[:4]])
fail=[s for s in ss if b[s]<0.9]; print("failed baseline seeds", fail, "others mean diff", np.mean([d[s]-b[s] for s in ss if s not in fail]).round(3))
for reg,lam in (("match",1.0),("rank",0.1)):
    a=acc(reg,lam); s2=sorted(set(a)&set(b)); df=np.array([a[s]-b[s] for s in s2]); sf=[abs(np.sum(df*np.array(sg)))/len(df) for sg in itertools.product([-1,1],repeat=len(df))]
    print(reg,len(s2),"mean diff",df.mean().round(3),"signflip p",np.mean(np.array(sf)>=abs(df.mean())-1e-12).round(3))
# escape
for reg,lam in (("none",0.0),("dec",1.0)):
    e=[D.mq_escape(D.MQ[(reg,lam,s)][1]) for s in D.mq_seeds(reg,lam)]; print(reg,"escaped by 1000:",sum(1 for x in e if x is not None and x<=1000),"of",len(e), "never", sum(x is None for x in e))
# L0 vs L96
learned=[(k,r) for k,(f,r) in D.MQ.items() if D.mq_acc(r)>0.5]
print("runs learned(acc>0.5)",len(learned),"of",len(D.MQ), "max |acc(L=0)-acc(L=96)| mean over n:", max(abs(np.mean([r['grid'][f'{n}_0'] for n in (8,16,24,32)])-np.mean([r['grid'][f'{n}_96'] for n in (8,16,24,32)])) for k,r in learned))
G=D.mq_geometry(); xs=[];ys=[];v1=[];mn=[];v0=[]
for k,arr in G.items():
    a=D.mq_acc(D.MQ[k][1]); ys.append(a); xs.append(np.median(arr[1][:,0])); v1.append(np.median(arr[1][:,1])); mn.append(min(np.median(arr[0][:,0]),np.median(arr[1][:,0]))); v0.append(np.median(arr[0][:,1]))
sp=lambda x,y: np.corrcoef(D.rk(np.array(x)),D.rk(np.array(y)))[0,1]
print("spearman acc vs key rank L1",round(sp(xs,ys),2),"min layer",round(sp(mn,ys),2),"value rank L1",round(sp(v1,ys),2),"value min?",round(sp([min(a,b) for a,b in zip(v0,v1)],ys),2), "n runs",len(ys))
