### matplotlib package -- https://matplotlib.org/3.5.0/index.html ###
from matplotlib import pyplot as plt    #   for plots -- https://matplotlib.org/3.5.0/api/pyplot_summary.html
from matplotlib import rc               #   for rcParams -- https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.rc.html 
 
### numpy package -- https://numpy.org/doc/stable/ ###
import numpy as np                      #   for general scientific computing
 
### scipy package -- https://docs.scipy.org/doc/scipy/reference/index.html ###
from scipy import constants as const    #   for physical constants -- https://docs.scipy.org/doc/scipy/reference/constants.html 
from scipy import optimize as opt       #   for optimization and fit -- https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
from scipy import special as sp         #   for special mathematical functions -- https://docs.scipy.org/doc/scipy/reference/tutorial/special.html
 
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':'20'})
## for 'Latin Modern' and other serif fonts use:
rc('font',**{'family':'serif','serif':['Latin Modern'], 'size':'16'})
rc('text', usetex=True)

def main():
    i = 1
    for i in range(1,5):
        
        DATA_DIR = f"../data/data20-gain30-0{i}.csv"

        data = np.loadtxt(DATA_DIR, delimiter=",", skiprows=2)

        time = np.array(data[:,0])
        voltage_1 = np.array(data[:,1])
        voltage_3 = np.array(data[:,3])
        
        fig, ax = plt.subplots()

        ax.plot(time, voltage_1, label = f'$Res$', color = 'tab:orange')
        ax.plot(time, voltage_3, label = f'$Rb$', color = 'tab:blue')

        ax.legend(loc = 'lower left')
        ax.set_xlabel('Zeit $t$ in s')
        ax.set_ylabel('Spannung $U$ in mV')
        ax.set_title(f'Gain30 in dB (0{i})')
        ax.grid(True)

        #plt.show()

        fig.savefig(f"../report/figures/plots/PNG/plot-data20-gain30-0{i}.png", format = 'png', bbox_inches = 'tight', dpi = 400)
        #fig.savefig(f"../report/figures/plots/EPS/plot-data20-gain30-0{i}.eps", format = 'eps', bbox_inches = 'tight')
        fig.savefig(f"../report/figures/plots/PDF/plot-data20-gain30-0{i}.pdf", format = 'pdf', bbox_inches = 'tight')
        #tikplotlib.save(f"../report/figures/tikz/plot_data20-gain30-0{i}.tex")


if __name__ == "__main__":
    main()
