import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke
import PIL
from PIL import Image
from scipy.ndimage import zoom


def plot_log_image(img, vmin = 0.01, vmax = 1e4):
    colormap = cm.inferno
    colormap.set_bad(color='black') # set color for NaN value to black
    plt.imshow(img, origin = 'lower', cmap = colormap, 
           norm=LogNorm(vmin, vmax)) #display range from vmin to vmax in log10 scale
    plt.colorbar()

def plot_linear_image(img, vmin = 0, vmax = 1):
    colormap = cm.inferno
    colormap.set_bad(color='black') # set color for NaN value to black
    plt.imshow(img, origin = 'lower', cmap = colormap, vmin = vmin, vmax = vmax)
    plt.colorbar()

def subplot_log_image(ax, img, vmin = 0.01, vmax = 1e4):
    colormap = cm.inferno
    colormap.set_bad(color='black') # set color for NaN value to black
    ax.imshow(img, origin = 'lower', cmap = colormap, norm=LogNorm(vmin, vmax))

def subplot_linear_image(ax, img, vmin=0, vmax=1):
    colormap = cm.inferno
    colormap.set_bad(color='black') # set color for NaN value to black
    ax.imshow(img, origin = 'lower', cmap = colormap, vmin = vmin, vmax = vmax)

def plot_vae_processed_image(subplot_imgs):
    colors0 = plt.cm.inferno(np.linspace(0, 0.55, 20*2))
    colors1 = plt.cm.inferno(np.linspace(0.55, 1, 25*2))
    colors2 = plt.cm.afmhot(np.linspace(0.8, 1, 10*2))
    mycolors = np.vstack((colors0, colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', mycolors)
    cmap =  mymap#matplotlib.cm.get_cmap('inferno')
    cmap.set_bad(color='k')
    subplot_1_img = subplot_imgs[0]
    subplot_2_img = subplot_imgs[1]
    subplot_3_img = subplot_imgs[2]
    subplot_4_img = subplot_imgs[3]

    plt.figure(figsize=[10, 10.08])
    count_row = 2
    count_column = 2
    lim_view = 1.7

    # Subplot 1
    ext = np.array([1, -1, -1, 1])/2*subplot_1_img.shape[0]*12.25e-3
    ax_this = plt.subplot(count_row, count_column, 1)
    plt.imshow(subplot_1_img, origin= 'lower', interpolation='nearest', cmap = None, #vmin = vmin, vmax = vmax, 
            extent=ext, norm=LogNorm(vmin=1e1, vmax=1e4))
    # plt.xlim([lim_view, -lim_view])
    # plt.ylim([-lim_view, lim_view])
    plt.axis('off')

    # left bottom text
    at = AnchoredText('(a)', loc=3, prop=dict(size=12, color = 'w'), pad=0., borderpad=0.5, frameon=False)
    ax_this.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="k", linewidth=2)])

    # right bottom circle and text
    # plt.gca().add_patch(matplotlib.patches.Circle((-1.2, -1.15), 0.10, color = 'None', hatch = None, ec = '0.79'))
    plt.text(-1.2, -1.4, 'V_V351_Ori', color = 'w', ha = 'center', va = 'center', fontsize = 8)

    # color bar
    cbar1 = plt.colorbar(orientation = 'vertical',  extend='both', 
                         shrink = 0.5, anchor = (0.7, 0.9),
                         use_gridspec = False)
    cbar1.ax.minorticks_on()
    cbar1.ax.tick_params(left = 0, bottom = 0, top = 0, right = 1, 
                    labelleft = 0, labelright = 1, labeltop = 0, labelbottom = 0,
                         which = 'both', labelsize=10, colors = 'w', direction = 'in')#, pad = -0.6)
    cbar1.set_ticks([10, 100, 1000, 10000], labels = [r'$10$', r'$10^2$', r'$10^3$', r'$10^4$'])

    # Subplot 2
    ext = np.array([1, -1, -1, 1])/2*subplot_2_img.shape[0]*12.25e-3
    ax_this = plt.subplot(count_row, count_column, 2)
    plt.imshow(subplot_2_img, origin= 'lower', interpolation='nearest', cmap = cmap, #vmin = vmin, vmax = vmax, 
            extent=ext, norm=LogNorm(vmin=1e1, vmax=1e4))
    # plt.xlim([lim_view, -lim_view])
    # plt.ylim([-lim_view, lim_view])
    plt.axis('off')

    plt.plot(0, 0, 'w.', ms = 1)#, markeredgecolor = '0.5')#, ha = 'center', va = 'center', color = '0.7')
    at = AnchoredText('(b)', loc=3, prop=dict(size=12, color = 'w'), pad=0., borderpad=0.5, frameon=False)
    ax_this.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="k", linewidth=2)])
    # plt.xlabel('$\Delta$RA', labelpad = -2)
    # ax_this.annotate("b",
    #             xy=(1.6, 0.7), #xycoords='data',
    #             xytext=(1.5, 1), #textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = 'w'), color = 'w')
    # ax_this.annotate("c",
    #             xy=(-0.5, -0.5), #xycoords='data',
    #             xytext=(-0.7, -0.8), #textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = 'w'), color = 'w')
    # ax_this.annotate("d",
    #             xy=(-0.5, 0.9), #xycoords='data',
    #             xytext=(-0.6, 1.2), #textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = 'w'), color = 'w')
    # ax_this.annotate("e",
    #             xy=(-0.4, 0.1), #xycoords='data',
    #             xytext=(-0.7, 0.2), #textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = 'w'), color = 'w')
    # plt.gca().add_patch(matplotlib.patches.Circle((-1.2, -1.15), 0.1, color = 'None', hatch = None, ec = '0.79'))
    cbar2 = plt.colorbar(orientation = 'vertical',  extend='both', 
                         shrink = 0.5, anchor = (0, 0.9),
                         use_gridspec = False)
    cbar2.ax.minorticks_on()
    cbar2.ax.tick_params(left = 0, bottom = 0, top = 0, right = 1, 
                    labelleft = 0, labelright = 1, labeltop = 0, labelbottom = 0,
                         which = 'both', labelsize=10, colors = 'w', direction = 'in')#, pad = -0.6)
    cbar2.set_ticks([10, 100, 1000, 10000], labels = [r'$10$', r'$10^2$', r'$10^3$', r'$10^4$'])
    plt.text(-1.2, -1.4, 'Preprocessed', color = 'w', ha = 'center', va = 'center', fontsize = 8)

    # Subplot 3
    # # lim_view = 0.6
    ext = np.array([1, -1, -1, 1])/2*subplot_3_img.shape[0]*12.25e-3
    ax_this = plt.subplot(count_row, count_column, 3)
    plt.imshow(subplot_3_img, origin= 'lower', interpolation='nearest', cmap = cmap, vmin = 0, vmax = 5, 
            extent=ext, #norm=LogNorm(vmin=0.05, vmax=1e2)
              )
    # plt.xlim([lim_view, -lim_view])
    # plt.ylim([-lim_view, lim_view])
    plt.axis('off')
    plt.plot(0, 0, 'w.', ms = 1)#, markeredgecolor = '0.5')#, ha = 'center', va = 'center', color = '0.7')
    at = AnchoredText('(c)', loc=3, prop=dict(size=12, color = 'w'), pad=0., borderpad=0.5, frameon=False)
    ax_this.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="k", linewidth=2)])
    # plt.gca().add_patch(matplotlib.patches.Circle((-1.2, -1.15), 0.1, color = 'None', hatch = None, ec = '0.79'))
    plt.text(-1.2, -1.4, 'Imputed \n Background', color = 'w', ha = 'center', va = 'center', fontsize = 8)

    cbar3 = plt.colorbar(orientation = 'vertical',  extend='both', 
                         shrink = 0.5, anchor = (0.7, 0.9),
                         use_gridspec = False)
    cbar3.ax.minorticks_on()
    cbar3.ax.tick_params(left = 0, bottom = 0, top = 0, right = 1, 
                    labelleft = 0, labelright = 1, labeltop = 0, labelbottom = 0,
                         which = 'both', labelsize=10, colors = 'w', direction = 'in')#, pad = -1)
    plt.subplots_adjust(wspace=0, hspace=0)
    cbar3.set_ticks([0, 2.5, 5], labels = [r'$0$',r'$2.5$',  r'$5$'])

    # Subplot 4
    # # lim_view = 0.6
    ext = np.array([1, -1, -1, 1])/2*subplot_4_img.shape[0]*12.25e-3
    ax_this = plt.subplot(count_row, count_column, 4)
    plt.imshow(subplot_4_img, origin= 'lower', interpolation='nearest', cmap = cmap, vmin = 0, vmax = 5, 
            extent=ext #, norm=LogNorm(vmin=1e1, vmax=1e5)
              )
    # plt.xlim([lim_view, -lim_view])
    # plt.ylim([-lim_view, lim_view])
    plt.axis('off')
    plt.plot(0, 0, 'w.', ms = 1)#, markeredgecolor = '0.5')#, ha = 'center', va = 'center', color = '0.7')
    at = AnchoredText('(d)', loc=3, prop=dict(size=12, color = 'w'), pad=0., borderpad=0.5, frameon=False)
    ax_this.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="k", linewidth=2)])
    # plt.gca().add_patch(matplotlib.patches.Circle((-1.2, -1.15), 0.1, color = 'None', hatch = None, ec = '0.79'))
    plt.text(-1.2, -1.4, 'Disk w/o \n Background', color = 'w', ha = 'center', va = 'center', fontsize = 8)

    cbar4 = plt.colorbar(orientation = 'vertical',  extend='both', 
                         shrink = 0.5, anchor = (0.1, 0.74),
                         use_gridspec = False)
    cbar4.ax.minorticks_on()
    cbar4.ax.tick_params(left = 0, bottom = 0, top = 0, right = 1, 
                    labelleft = 0, labelright = 1, labeltop = 0, labelbottom = 0,
                         which = 'both', labelsize=10, colors = 'w', direction = 'in')#, pad = -0.6)
    cbar4.set_ticks([0, 2.5, 5], labels = [r'$0$', r'$2.5$', r'$5$'])
    plt.subplots_adjust(wspace=0, hspace=0)

    #plt.savefig('plot_examples\\images\\vae_disk_v1.png', bbox_inches = 'tight',
    #    pad_inches = 0.01, dpi = 600, transparent=True)
    #plt.savefig("plot_examples\\images\\vae_disk_v1.pdf")
    # fig.savefig('plot_examples\\images\\data_processing_examples.png')
    