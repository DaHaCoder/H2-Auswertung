### matplotlib package -- https://matplotlib.org/4.5.0/index.html ###
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

def find_local_maxima(thresh: float, time: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    local_maxima = []
    new_maximum = True

    for i, v in enumerate(voltage):
        if new_maximum and v > thresh:
            local_maxima.append(i)
            new_maximum = False
        elif not new_maximum:
            if v > voltage[local_maxima[-1]]:
                local_maxima[-1] = i
            elif v <= thresh:
                new_maximum = True

    return local_maxima

def main():
    DATA_DIR = "../data/data00.csv"
    
    data = np.loadtxt(DATA_DIR, delimiter=",", skiprows=2)

    time = np.array(data[:,0])
    voltage_1 = np.array(data[:,1])

    local_maxima = find_local_maxima(3e-2, time, voltage_1)

    time_distances = []
    
    for i in range(len(time[local_maxima])-1):
        t1 = time[local_maxima][i]
        t2 = time[local_maxima][i+1]

        delta_t = t2 - t1
        time_distances.append(delta_t)
    
    print("======================")
    print("ARRAY_time_distances = ", np.array(time_distances))

    mean_delta_t = 1/len(np.array(time_distances))*np.sum(np.array(time_distances))
    
    print("======================")
    print("mean_delta_t = ", mean_delta_t)

    sigma_delta_t = np.sqrt(1/len(np.array(time_distances))*np.sum(np.power(np.array(time_distances) - mean_delta_t, 2)))

    print("======================")
    print("sigma_delta_t = ", sigma_delta_t)

    standard_error_delta_t = sigma_delta_t/np.sqrt(len(np.array(time_distances)))

    print("======================")
    print("standard_error_delta_t = ", standard_error_delta_t)


    d = 0.1
    c = const.c


    def time_to_freq(t, c, d, mean_delta_t):
        return c/(4.0*d)*1/mean_delta_t*t

    def freq_to_time(nu, c, d, mean_delta_t):
        return (4.0*d)/c*mean_delta_t*nu


    fig, ax = plt.subplots()
    
    ax.vlines(time[local_maxima], 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:green', linestyles = 'dashed', linewidth = 1)
    
    ax.plot(time, voltage_1, label = 'Res', color = 'tab:orange')
    
    ax.set_xlabel(r'Zeit $t$ in s')
    ax.set_ylabel('Spannung $U$ in mV')
    
    secax = ax.secondary_xaxis('top', functions = (time_to_freq, freq_to_time))
    secax.set_xlabel(r'Frequenz $\nu$ in Hz')
    secax.ticklabel_format(style = 'sci', axis = 'secax', scilimits = (0,0))

    ax.set_ylim(0,0.07)
    ax.grid(True)

    fig.savefig("../report/figures/plots/PNG/plot-data00-resonator.png", format = 'png', bbox_inches = 'tight', dpi = 400)
    #fig_i.savefig("../report/figures/plots/EPS/plot-data00-resonator.eps", format = 'eps', bbox_inches = 'tight')
    fig.savefig("../report/figures/plots/PDF/plot-data00-resonator.pdf", format = 'pdf', bbox_inches = 'tight')
    #plt.show()
    #tikplotlib.save("../report/figures/tikz/plot-data00-resonator.tex")
    

if __name__ == "__main__":
    main()
