'''
Shamelessly stealing this code from Harshil, so we have the same plot appearance.
'''

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

#sns.palplot(sns.cubehelix_palette(18, start=2, rot=0.1, dark=0, light=0.99))
cold_cmap = sns.cubehelix_palette(18, start=2, rot=0.1, dark=0, light=0.99, as_cmap=True)

sns.set_style("white")
sns.set_style("ticks")

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

#main plotting function; creates both a hexbin and violin plot
def genplots_M(b, mse, annotate=False, save=False, string=None, plot_type='M'):
    #fig = plt.figure()
    plt.clf()
    plt.close()
    #plt.subplot(121)

    g = sns.JointGrid(b[:,0],b[:,1])
    g.plot_marginals(sns.distplot, color=".5")
    g.plot_joint(plt.hexbin, bins='log', gridsize=30, cmap=cold_cmap, extent=[0, np.max(b[:,0]), 0, np.max(b[:,0])])
    a=np.linspace(0,max(b[:,0]),20)
    plt.plot(a,a,'k--')
    if annotate:
        plt.annotate('$MSE: %.6f $' % mse, xy=(0.70, 0.95), xycoords='axes fraction')
        g.annotate(pearsonr, template="{stat} = {val:.3f} (p = {p:.3g})", loc='best')
    if plot_type=='M':
        plt.xlabel('$M_{SAM} (10^{10} M_{\odot}/h)$', fontsize=16)
        plt.ylabel('$M_{predicted} (10^{10} M_{\odot}/h)$', fontsize=16)
    else:
        plt.xlabel('$R_{SAM} (Mpc/h)$', fontsize=16)
        plt.ylabel('$R_{predicted} (Mpc/h)$', fontsize=16)
    plt.xlim([0,np.max(b[:,0])])
    plt.ylim([0,np.max(b[:,0])])
    cax = g.fig.add_axes([1, 0.20, .01, 0.5])
    cb = plt.colorbar(cax=cax)
    cb.set_label('$\log_{10}(\mathcal{N})$')
    if save:
        plt.savefig(string + '1.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    #ax2 = fig.add_subplot(121)
    #plt.subplot(122)

    if plot_type=='M':
        sns.violinplot(b,names=('$M_{SAM}$', '$M_{predicted}$'), bw='scott', gridsize=1000, color='PuBu')
    else:
        sns.violinplot(b,names=('$R_{SAM}$', '$R_{predicted}$'), bw='scott', gridsize=1000, color='PuBu')
    sns.despine(bottom=True)
    plt.tick_params(axis='x', which='both', bottom='off', top='off')
    plt.ylim([0,(np.mean(b[:,1])*6)])
    if plot_type=='M':
        plt.ylabel('$M (10^{10} M_{\odot}/h)$', fontsize=13)
    else:
        plt.ylabel('$R (Mpc/h)$', fontsize=13)
    if save:
        plt.savefig(string + '2.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#calculate mse
def mse(test, pred):
    s=0.0
    if isinstance(pred, float):
        for i in xrange(len(test)):
            s+=(test[i]-pred)**2
    else:
        for i in xrange(len(test)):
            s+=(test[i]-pred[i])**2
    return s/(len(test))

#plot distribution using a violinplot
def plot_distribution(mass, string, color):
    sns.violinplot(mass, bw='silverman', gridsize=1000, color=color)
    sns.despine(top=True, right=True, left=False, bottom=True)
    plt.ylim([0,(np.mean(mass)*8)])
    plt.xlabel(string)
    plt.show()

#plot the stellar mass-halo mass relation
def plot_smhm(stellar_sam, stellar_predicted, halo_np, save=False):
    figure = plt.figure(figsize=(8,8))
    mh = halo_np*0.086
    frac_sam = stellar_sam/mh
    frac_ml = stellar_predicted/mh
    mh = halo_np*8.6*10**8

    bins = np.logspace(11.0, 15.0, 75)
    n, _ = np.histogram(mh, bins=bins)
    sy, _ = np.histogram(mh, bins=bins, weights=frac_sam)
    sy2, _ = np.histogram(mh, bins=bins, weights=frac_sam*frac_sam)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, fmt='-', color='blue', label='G11', linewidth=2)
    plt.fill_between((_[1:] + _[:-1])/2, mean-std, mean+std, alpha=0.07, color='blue')

    n, _ = np.histogram(mh, bins=bins)
    sy, _ = np.histogram(mh, bins=bins, weights=frac_ml)
    sy2, _ = np.histogram(mh, bins=bins, weights=frac_ml*frac_ml)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, fmt='-',color='green',label='Predicted', linewidth=2)

    plt.fill_between((_[1:] + _[:-1])/2, mean-std, mean+std, alpha=.25, color='green')
    plt.legend(loc='best')

    #plt.xscale('log', nonposx='clip')
    plt.yscale('log', nonposy='clip')

    plt.xlabel('$M_{h} (M_{\odot})$', fontsize=16)
    plt.ylabel('$M_{\star}/M_{h} (M_{\odot})$', fontsize=16)

    plt.ylim([10**(-3.3), 10**(-1)])

    if save:
        plt.savefig('smhm.pdf', bbox_inches='tight')

    plt.show()

#plot the average cold gas mass fraction as a function of stellar mass
def plot_coldgasfrac(cold_sam, cold_predicted, stellar_sam, stellar_predicted, save=False):
    figure = plt.figure()

    frac_sam = cold_sam/stellar_sam
    bins = np.logspace(9.5, 12.0, 75)
    n, _ = np.histogram(stellar_sam*10**10, bins=bins)
    sy, _ = np.histogram(stellar_sam*10**10, bins=bins, weights=frac_sam)
    sy2, _ = np.histogram(stellar_sam*10**10, bins=bins, weights=frac_sam*frac_sam)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, fmt='k-', label='G11')

    frac_ml = cold_predicted/stellar_predicted
    n, _ = np.histogram(stellar_predicted*10**10, bins=bins)
    sy, _ = np.histogram(stellar_predicted*10**10, bins=bins, weights=frac_ml)
    sy2, _ = np.histogram(stellar_predicted*10**10, bins=bins, weights=frac_ml*frac_ml)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, fmt='b-', label='Predicted', color='gray')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim([10**10, 10**12])

    plt.xlabel('$M_{\star} (M_{\odot})$',fontsize=16)
    plt.ylabel('$M_{gas}/M_{\star}$', fontsize=16)
    plt.legend(loc='best', prop={'size':12})

    if save:
        plt.savefig('gasfrac.pdf', bbox_inches='tight')

    plt.show()

#plot the black hole mass-bulge mass relation
def plot_bhbulge(bh_sam, bh_predicted, bulge_sam, bulge_predicted, save=False):
    figure = plt.figure(figsize=(8,8))
    bins = np.logspace(9.0, 13.0, 100)
    y = bh_sam*10**10
    n, _ = np.histogram(bulge_sam*10**10, bins=bins)
    sy, _ = np.histogram(bulge_sam*10**10, bins=bins, weights=y)
    sy2, _ = np.histogram(bulge_sam*10**10, bins=bins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, fmt='k-', label='G11')
    plt.fill_between((_[1:] + _[:-1])/2, mean-std, mean+std, alpha=0.07, color='blue')

    y = bh_predicted*10**10
    n, _ = np.histogram(bulge_predicted*10**10, bins=bins)
    sy, _ = np.histogram(bulge_predicted*10**10, bins=bins, weights=y)
    sy2, _ = np.histogram(bulge_predicted*10**10, bins=bins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    plt.errorbar((_[1:] + _[:-1])/2, mean, fmt='b-', label='Predicted', color='green')
    plt.fill_between((_[1:] + _[:-1])/2, mean-std, mean+std, alpha=.25, color='green')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim([10**9.0, max(bulge_sam*10**10)])
    plt.ylim([10**6.8, 10**10])

    plt.xlabel('$M_{Bulge} (M_{\odot})$', fontsize=16)
    plt.ylabel('$M_{BH} (M_{\odot})$', fontsize=16)

    plt.legend(loc='best', prop={'size':10})

    if save:
        plt.savefig('bhbulge.pdf', bbox_inches='tight')

    plt.show()