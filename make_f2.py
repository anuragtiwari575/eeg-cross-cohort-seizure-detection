"""
=============================================================================
F2 -- GENERALISATION ACROSS FOURTEEN ARCHITECTURES
=============================================================================
    python -u make_f2.py

Writes F2_architectures.pdf and .png to D:\\figures.

WHY THIS IS A SEPARATE SCRIPT

F2 comes from the first sweep, which ran fourteen architectures at one seed
each. Its results file was lost when the drive letter changed, but the
numbers themselves were preserved in the notebook log, so they are embedded
below rather than reread from disk. The later sweep on this machine holds
four architectures at five seeds and supports the seed analysis, not this
figure; regenerating F2 from it would show four points where the manuscript
reports fourteen.

The handcrafted reference lines are the matched-window values reported in
the text (0.709 at S2, 0.684 at S3), which is the comparison the figure is
making.
=============================================================================
"""
from pathlib import Path
import io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

FIG = Path(r"D:\figures"); FIG.mkdir(parents=True, exist_ok=True)
LIN_S2, LIN_S3 = 0.709, 0.684

DATA = """model,setting,auroc
EEGNet,S1,0.9878571702038057
EEGNet,S1,0.9021765825505004
EEGNet,S1,0.9495088245088246
EEGNet,S1,0.9077955538533851
EEGNet,S2,0.8467672384487775
EEGNet,S2,0.7452196937169029
EEGNet,S2,0.8210320168330063
EEGNet,S2,0.8900679623406895
EEGNet,S2,0.9479625816619639
EEGNet,S3,0.8167532305646736
EEGNet,S3,0.8731515955062815
EEGITNet,S1,0.9884824972243138
EEGITNet,S1,0.9417173205624118
EEGITNet,S1,0.961517649017649
EEGITNet,S1,0.8364552687011472
EEGITNet,S2,0.8730910568110739
EEGITNet,S2,0.8067596734950161
EEGITNet,S2,0.8451586827685096
EEGITNet,S2,0.9124151605969788
EEGITNet,S2,0.9516334934797529
EEGITNet,S3,0.8353627373582083
EEGITNet,S3,0.8486558056869955
EEGTCNet,S1,0.9867979428017203
EEGTCNet,S1,0.8091729600294713
EEGTCNet,S1,0.9521728271728271
EEGTCNet,S1,0.543991434987727
EEGTCNet,S2,0.872414997552419
EEGTCNet,S2,0.7991206018550581
EEGTCNet,S2,0.8231041542267338
EEGTCNet,S2,0.9276025489661852
EEGTCNet,S2,0.9537783115991765
EEGTCNet,S3,0.8530158862065622
EEGTCNet,S3,0.8772019171314668
SincShallowNet,S1,0.9487646600900983
SincShallowNet,S1,0.9387701848099711
SincShallowNet,S1,0.957958707958708
SincShallowNet,S1,0.44222098428006895
SincShallowNet,S2,0.8607233791574883
SincShallowNet,S2,0.8260124793972216
SincShallowNet,S2,0.852490525340905
SincShallowNet,S2,0.9295542336451428
SincShallowNet,S2,0.938226823254277
SincShallowNet,S3,0.8144949515934682
SincShallowNet,S3,0.8475103619939882
SCCNet,S1,0.9864150895238581
SCCNet,S1,0.9488242156320992
SCCNet,S1,0.9771062271062271
SCCNet,S1,0.9169611614992254
SCCNet,S2,0.8586110657855375
SCCNet,S2,0.7920696039187985
SCCNet,S2,0.7984554173115122
SCCNet,S2,0.9103322435140617
SCCNet,S2,0.9195767558911004
SCCNet,S3,0.8376410991537513
SCCNet,S3,0.7816060721583565
EEGInceptionERP,S1,0.986705419926237
EEGInceptionERP,S1,0.9245870940013506
EEGInceptionERP,S1,0.9809773559773559
EEGInceptionERP,S1,0.9191372316905453
EEGInceptionERP,S2,0.7571521205569607
EEGInceptionERP,S2,0.8689752166465832
EEGInceptionERP,S2,0.6777072340494672
EEGInceptionERP,S2,0.9468614718614718
EEGInceptionERP,S2,0.9244012303312235
EEGInceptionERP,S3,0.8062190966884115
EEGInceptionERP,S3,0.7628890691025745
ShallowFBCSPNet,S1,0.9872222718513508
ShallowFBCSPNet,S1,0.9552403757598085
ShallowFBCSPNet,S1,0.9696969696969696
ShallowFBCSPNet,S1,0.9326114583151994
ShallowFBCSPNet,S2,0.8660942896576107
ShallowFBCSPNet,S2,0.7898920576000591
ShallowFBCSPNet,S2,0.8155078742236467
ShallowFBCSPNet,S2,0.9243227984137075
ShallowFBCSPNet,S2,0.9320208952947457
ShallowFBCSPNet,S3,0.7989605407127325
ShallowFBCSPNet,S3,0.8236242389984993
EEGNeX,S1,0.9922089357955053
EEGNeX,S1,0.9879658623442009
EEGNeX,S1,0.9865759240759241
EEGNeX,S1,0.9457026965861811
EEGNeX,S2,0.8687938524924532
EEGNeX,S2,0.7482989145740707
EEGNeX,S2,0.8030156428348018
EEGNeX,S2,0.8904960191323827
EEGNeX,S2,0.9428680190142098
EEGNeX,S3,0.8246360853727788
EEGNeX,S3,0.8359437895001038
ATCNet,S1,0.9863321379803214
ATCNet,S1,0.9382022471910113
ATCNet,S1,0.9647852147852148
ATCNet,S1,0.923471963511655
ATCNet,S2,0.8822661710315194
ATCNet,S2,0.8199219056588964
ATCNet,S2,0.8510818171847772
ATCNet,S2,0.9266135379771743
ATCNet,S2,0.9576234779735122
ATCNet,S3,0.822870181853437
ATCNet,S3,0.8529539679995332
TIDNet,S1,0.9864852792914662
TIDNet,S1,0.9117240744151778
TIDNet,S1,0.9408091908091909
TIDNet,S1,0.6416969865779991
TIDNet,S2,0.7724136377852111
TIDNet,S2,0.7902727184587044
TIDNet,S2,0.8379581449572067
TIDNet,S2,0.923686162322526
TIDNet,S2,0.9357007041358448
TIDNet,S3,0.8070237946842842
TIDNet,S3,0.8617354964599963
Deep4Net,S1,0.9789366888296176
Deep4Net,S1,0.9345029778350832
Deep4Net,S1,0.9108807858807858
Deep4Net,S1,0.8900214125306827
Deep4Net,S2,0.7941079587174676
Deep4Net,S2,0.6706385591674862
Deep4Net,S2,0.7604148133705413
Deep4Net,S2,0.9058741258741259
Deep4Net,S2,0.9482701644678309
Deep4Net,S3,0.7709145302696638
Deep4Net,S3,0.8647791904866861
EEGSimpleConv,S1,0.9891205860207507
EEGSimpleConv,S1,0.8838951310861423
EEGSimpleConv,S1,0.9616633366633367
EEGSimpleConv,S1,0.9411242449036435
EEGSimpleConv,S2,0.9008130558048462
EEGSimpleConv,S2,0.796440278489545
EEGSimpleConv,S2,0.8010116458081997
EEGSimpleConv,S2,0.9082092150273968
EEGSimpleConv,S2,0.9586105391596126
EEGSimpleConv,S3,0.8222974751042229
EEGSimpleConv,S3,0.8329199257163405
EEGConformer,S1,0.9880262637348614
EEGConformer,S1,0.9090071836433966
EEGConformer,S1,0.9792707292707292
EEGConformer,S1,0.9342826802221332
EEGConformer,S2,0.8403262761415247
EEGConformer,S2,0.8533591185473482
EEGConformer,S2,0.7900905017811953
EEGConformer,S2,0.9054949595858687
EEGConformer,S2,0.9498462085970665
EEGConformer,S3,0.8686746028448359
EEGConformer,S3,0.828537389301283
SPARCNet,S1,0.9857578580635282
SPARCNet,S1,0.9345490268312151
SPARCNet,S1,0.9612678987678988
SPARCNet,S1,0.8145727068572324
SPARCNet,S2,0.8933956956569036
SPARCNet,S2,0.8137797384081958
SPARCNet,S2,0.6995875019802342
SPARCNet,S2,0.9248315321042594
SPARCNet,S2,0.9677367243702178
SPARCNet,S3,0.8179377709450008
SPARCNet,S3,0.8775748030333548
"""

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "red": "#D55E00",
     "grey": "#999999", "sky": "#56B4E9"}

r = pd.read_csv(io.StringIO(DATA))
piv = r.pivot_table(index="model", columns="setting",
                    values="auroc", aggfunc="mean")[["S1", "S2", "S3"]]
piv = piv.sort_values("S2")
d1 = (piv.S1 - piv.S2).round(3)
d2 = (piv.S2 - piv.S3).round(3)
w = wilcoxon(d1, d2)

print("%d architectures" % len(piv))
print(piv.round(3).to_string())
print("\nmean S1->S2 %.3f | mean S2->S3 %.3f | ratio %.0f%%"
      % (d1.mean(), d2.mean(), 100 * d2.mean() / d1.mean()))
print("Wilcoxon p = %.4f | direction %d/%d"
      % (w.pvalue, (d1 > d2).sum(), len(d1)))
print("S3 spread %.3f (%s %.3f to %s %.3f)"
      % (piv.S3.max() - piv.S3.min(), piv.S3.idxmin(), piv.S3.min(),
         piv.S3.idxmax(), piv.S3.max()))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3),
                         gridspec_kw=dict(width_ratios=[1.45, 1]))
ax = axes[0]
x = [0, 1, 2]
for m, row in piv.iterrows():
    ax.plot(x, [row.S1, row.S2, row.S3], marker="o", ms=3.5,
            color=C["blue"], alpha=.5, lw=1.0)
    ax.annotate(m, (2.05, row.S3), va="center", fontsize=5.8, color="#333")
ax.plot(x, [piv.S1.mean(), piv.S2.mean(), piv.S3.mean()], marker="s", ms=5,
        color=C["red"], lw=2.0, zorder=5, label="mean, learned representations")
ax.hlines([LIN_S2], .82, 1.18, color=C["orange"], lw=2.2, zorder=5)
ax.hlines([LIN_S3], 1.82, 2.18, color=C["orange"], lw=2.2, zorder=5)
ax.plot([], [], color=C["orange"], lw=2.2, label="handcrafted features")
ax.axhline(.5, color=C["grey"], ls=":", lw=.8)
ax.text(-.12, .505, "chance", fontsize=6, color=C["grey"], va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(["S1\nwithin-patient", "S2\ncross-patient", "S3\ncross-cohort"])
ax.set_ylabel("AUROC"); ax.set_xlim(-.20, 2.95); ax.set_ylim(.45, 1.0)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
ax.set_title("(a) generalisation by setting", loc="left")

ax = axes[1]
yy = np.arange(len(piv))
ax.barh(yy - .2, d1, height=.38, color=C["blue"],
        label="S1 $\\rightarrow$ S2  (new patient)")
ax.barh(yy + .2, d2, height=.38, color=C["sky"],
        label="S2 $\\rightarrow$ S3  (new cohort)")
ax.axvline(0, color="k", lw=.8)
ax.set_yticks(yy); ax.set_yticklabels(piv.index, fontsize=6)
ax.set_xlabel("AUROC drop"); ax.legend(loc="lower right")
ax.set_title("(b) cost of each boundary", loc="left")
ax.text(.98, .98,
        "mean drop\nS1$\\rightarrow$S2  %.3f\nS2$\\rightarrow$S3  %.3f\n"
        "Wilcoxon $p$ = %.3f\n($n$ = %d)" % (d1.mean(), d2.mean(), w.pvalue, len(d1)),
        transform=ax.transAxes, ha="right", va="top", fontsize=6,
        color="#333", linespacing=1.4)

fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(FIG / ("F2_architectures." + e))
plt.close(fig)
print("\nwrote F2_architectures.pdf and .png to %s" % FIG)
