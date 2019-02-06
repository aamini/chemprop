import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_path", help="processed visualization data", type=str, required=True)
args = parser.parse_args()

x, y, names, c = [], [], [], []
with open(args.data_path, 'r') as rf:
    rf.readline()
    for line in rf:
        line = line.strip().split(',')
        names.append(line[0])
        x.append(float(line[1]))
        y.append(float(line[2]))
        c.append(float(line[3]))

cmap = plt.cm.jet

fig,ax = plt.subplots(figsize=(12,12))
sc = plt.scatter(x,y,c=c, s=3, cmap=cmap)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"), fontsize=12)
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "\n".join([names[n] for n in ind["ind"]])
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()