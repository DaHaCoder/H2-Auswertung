## matplotlib package -- https://matplotlib.org/4.5.0/index.html ###
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

def lorentz_fit_func(omega, I_0, omega_0, gamma, I_y):
    return I_0/np.pi*(gamma/2.0)*1/((omega - omega_0)**2.0 + (gamma/2.0)**2) + I_y

def time_to_freq(t, c, d, mean_delta_t):
    return c/(4.0*d)*1/mean_delta_t*t

def freq_to_time(nu, c, d, mean_delta_t):
    return (4.0*d)/c*mean_delta_t*nu

def main():
    DATA_DIR = "../data/data20-gain30-04.csv"
    data = np.loadtxt(DATA_DIR, delimiter=",", skiprows=2)

    time = np.array(data[:,0])
    voltage_1 = np.array(data[:,1])
    voltage_3 = np.array(data[:,3])
    
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

    mean_delta_t = 1/len(time_distances)*np.sum(np.array(time_distances))

    d = 0.1
    c = const.c
    
    #   initial guess and mask for peak #1
    #   ==================================
    p0_1 = [0.000001, 0.016949, 0.0000001, 0.076]                             #   I_0, omega_0, gamma, I_y
    t_init_1 = 0.01693
    t_end_1 = 0.01696
    mask_1 = (time > t_init_1) & (time < t_end_1)
    
    #   initial guess and mask for peak #2
    #   ==================================
    p0_2 = [0.000000001, 0.016975, 0.000001, 0.076]                           #   I_0, omega_0, gamma, I_y
    t_init_2 = 0.016968
    t_end_2 = 0.01699
    mask_2 = (time > t_init_2) & (time < t_end_2)
    
    #   initial guess and mask for peak #3
    #   ==================================
    p0_3 = [0.000001, 0.017008, 0.0000001, 0.077]                             #   I_0, omega_0, gamma, I_y
    t_init_3 = 0.0169915
    t_end_3 = 0.01701308
    mask_3 = (time > t_init_3) & (time < t_end_3)
    
    #   initial guess and mask for peak #4
    #   ==================================
    p0_4 = [0.000001, 0.017026, 0.0000001, 0.077]                             #   I_0, omega_0, gamma, I_y
    t_init_4 = 0.017015
    t_end_4 = 0.01705
    mask_4 = (time > t_init_4) & (time < t_end_4)
    
    #   initial guess and mask for peak #5
    #   ==================================
    p0_5 = [0.000001, 0.017078, 0.0000001, 0.078]                             #   I_0, omega_0, gamma, I_y
    t_init_5 = 0.01706
    t_end_5 = 0.01710
    mask_5 = (time > t_init_5) & (time < t_end_5)
    
    
    ### ================= ###
    ### FIT FOR THE PEAKS ###
    ### ================= ###   

    LIST_omega_0 = []           #   list of values for fitted omega_0
    LIST_gamma = []             #   list of values for fitted gamma

    #   fit for peak #1
    #   ================
    popt_1, pcov_1 = opt.curve_fit(lorentz_fit_func, time[mask_1], voltage_3[mask_1], p0_1)
    I_0, omega_0, gamma, I_y = popt_1

    print("\n=== PARAMETERS FOR PEAK #1 ===")
    print("=============================")
    print("I_0, omega_0, gamma, I_y = ", popt_1)
    LIST_omega_0.append(popt_1[1])
    LIST_gamma.append(popt_1[2])
    
    time_new_1 = np.linspace(t_init_1, t_end_1, 1000)
    voltage_3_fit_peak_1 = lorentz_fit_func(time_new_1, *popt_1)
    
    #   fit for peak #2
    #   ================
    popt_2, pcov_2 = opt.curve_fit(lorentz_fit_func, time[mask_2], voltage_3[mask_2], p0_2)
    I_0, omega_0, gamma, I_y = popt_2

    print("\n=== PARAMETERS FOR PEAK #2 ===")
    print("=============================")
    print("I_0, omega_0, gamma, I_y = ", popt_2)
    LIST_omega_0.append(popt_2[1])
    LIST_gamma.append(popt_2[2])

    time_new_2 = np.linspace(t_init_2, t_end_2, 1000)
    voltage_3_fit_peak_2 = lorentz_fit_func(time_new_2, *popt_2)
    
    #   fit for peak #3
    #   ================
    popt_3, pcov_3 = opt.curve_fit(lorentz_fit_func, time[mask_3], voltage_3[mask_3], p0_3)
    I_0, omega_0, gamma, I_y = popt_3

    print("\n=== PARAMETERS FOR PEAK #3 ===")
    print("=============================")
    print("I_0, omega_0, gamma, I_y = ", popt_3)
    LIST_omega_0.append(popt_3[1])
    LIST_gamma.append(popt_3[2])

    time_new_3 = np.linspace(t_init_3, t_end_3, 1000)
    voltage_3_fit_peak_3 = lorentz_fit_func(time_new_3, *popt_3)
    
    #   fit for peak #4
    #   ================
    popt_4, pcov_4 = opt.curve_fit(lorentz_fit_func, time[mask_4], voltage_3[mask_4], p0_4)
    I_0, omega_0, gamma, I_y = popt_4

    print("\n=== PARAMETERS FOR PEAK #4 ===")
    print("=============================")
    print("I_0, omega_0, gamma, I_y = ", popt_4)
    LIST_omega_0.append(popt_4[1])
    LIST_gamma.append(popt_4[2])

    time_new_4 = np.linspace(t_init_4, t_end_4, 1000)
    voltage_3_fit_peak_4 = lorentz_fit_func(time_new_4, *popt_4)
    
    #   fit for peak #5
    #   ================
    popt_5, pcov_5 = opt.curve_fit(lorentz_fit_func, time[mask_5], voltage_3[mask_5], p0_5)
    I_0, omega_0, gamma, I_y = popt_5

    print("\n=== PARAMETERS FOR PEAK #5 ===")
    print("I_0, omega_0, gamma, I_y = ", popt_5)
    LIST_omega_0.append(popt_5[1])
    LIST_gamma.append(popt_5[2])

    time_new_5 = np.linspace(t_init_5, t_end_5, 1000)
    voltage_3_fit_peak_5 = lorentz_fit_func(time_new_5, *popt_5)
    
    
    ### ======== ###
    ### THE PLOT ###
    ### ======== ###
    
    fig, ax = plt.subplots()

    #   raw data plot in range of peaks
    #   ===============================

    ax.plot(time_to_freq(time, c, d, mean_delta_t), voltage_3, color = 'tab:blue') 
    #ax.plot(time, voltage_3, color = 'tab:blue') 
    
    xmin = 0.0169
    xmax = 0.01712
    ax.set_xlim(time_to_freq(xmin, c, d, mean_delta_t), time_to_freq(xmax, c, d, mean_delta_t))
    #ax.set_xlim(xmin, xmax)
    
    ymin = 0.0758
    ymax = 0.0815
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'Frequenz $\nu$ in Hz')
    ax.set_ylabel('Spannung $U$ in mV')
    ax.grid(True)
     
    #   plot fitted peak #1
    #   ===================

    ax.plot(time_to_freq(time_new_1, c, d, mean_delta_t), voltage_3_fit_peak_1, color = 'tab:orange')
    #ax.plot(time_new_1, voltage_3_fit_peak_1, color = 'tab:orange')
    
    #   plot fitted peak #2
    #   ===================

    ax.plot(time_to_freq(time_new_2, c, d, mean_delta_t), voltage_3_fit_peak_2, color = 'tab:orange')
    #ax.plot(time_new_2, voltage_3_fit_peak_2, color = 'tab:orange')
    
    #   plot fitted peak #3
    #   ===================

    ax.plot(time_to_freq(time_new_3, c, d, mean_delta_t), voltage_3_fit_peak_3, color = 'tab:orange')
    #ax.plot(time_new_3, voltage_3_fit_peak_3, color = 'tab:orange')

    #   plot fitted peak #4
    #   ===================
    
    ax.plot(time_to_freq(time_new_4, c, d, mean_delta_t), voltage_3_fit_peak_4, color = 'tab:orange')
    #ax.plot(time_new_4, voltage_3_fit_peak_4, color = 'tab:orange')

    #   plot fitted peak #5
    #   ===================

    ax.plot(time_to_freq(time_new_5, c, d, mean_delta_t), voltage_3_fit_peak_5, color = 'tab:orange')
    #ax.plot(time_new_5, voltage_3_fit_peak_5, color = 'tab:orange')
    
    #   plot vertical lines at position of peaks (omega_0)
    #   ==================================================  
    ax.vlines(time_to_freq(np.array(LIST_omega_0), c, d, mean_delta_t), 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:green', linestyles = 'dashed', linewidth = 1)
    #ax.vlines(time, 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:green', linestyles = 'dashed', linewidth = 1)
   
    
    ### ========= ###
    ### SAVE PLOT ###
    ### ========= ###
    
    fig.savefig("../report/figures/plots/PNG/plot-data20-gain30-04-rubidium-fit.png", format = 'png', bbox_inches = 'tight', dpi = 400)
    #fig_i.savefig("../report/figures/plots/EPS/plot-data20-gain30-04-rubidium-fit.eps", format = 'eps', bbox_inches = 'tight')
    fig.savefig("../report/figures/plots/PDF/plot-data20-gain30-04-rubidium-fit.pdf", format = 'pdf', bbox_inches = 'tight')
    #plt.show()
    #tikzplotlib.save("../report/figures/tikz/plot-data20-gain30-04-rubidium-fit.tex")
    
    
    ### ============================= ###
    ### CALCULATE FREQUENCY DISTANCES ### 
    ### ============================= ###

    #   distance to peak #1
    #   ===================

    print("\n=== PEAK 1, 2 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[1] - LIST_omega_0[0]), c, d, mean_delta_t)*10**(-6))
    
    print("\n=== PEAK 1, 3 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[2] - LIST_omega_0[0]), c, d, mean_delta_t)*10**(-6))
    
    print("\n=== PEAK 1, 4 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[3] - LIST_omega_0[0]), c, d, mean_delta_t)*10**(-6))
    
    print("\n=== PEAK 1, 5 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[4] - LIST_omega_0[0]), c, d, mean_delta_t)*10**(-6))
    
    #   distance to peak #2
    #   ===================

    print("\n=== PEAK 2, 3 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[2] - LIST_omega_0[1]), c, d, mean_delta_t)*10**(-6))
    
    print("\n=== PEAK 2, 4 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[3] - LIST_omega_0[1]), c, d, mean_delta_t)*10**(-6))
    
    print("\n=== PEAK 2, 5 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[4] - LIST_omega_0[1]), c, d, mean_delta_t)*10**(-6))

    #   distance to peak #3
    #   ===================

    print("\n=== PEAK 3, 4 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[3] - LIST_omega_0[2]), c, d, mean_delta_t)*10**(-6))
    
    print("\n=== PEAK 3, 5 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[4] - LIST_omega_0[2]), c, d, mean_delta_t)*10**(-6))
    
    #   distance to peak #4
    #   ===================

    print("\n=== PEAK 4, 5 ===")
    print("=================")
    print("freq_dist in MHz = ", time_to_freq((LIST_omega_0[4] - LIST_omega_0[3]), c, d, mean_delta_t)*10**(-6))
          
    
if __name__ == "__main__":
    main()
