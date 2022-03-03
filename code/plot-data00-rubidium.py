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


def lin_fit_func(t, a, U0):
    return a*t + U0

def gauss_fit_func(omega, I_0, omega_0, delta_omega, I_y):
    return I_0*np.exp(-((omega - omega_0)/(delta_omega/(2.0*np.sqrt(np.log(2.0)))))**2.0) + I_y

def time_to_freq(t, c, d, mean_delta_t):
    return c/(4.0*d)*1/mean_delta_t*t

def freq_to_time(nu, c, d, mean_delta_t):
    return (4.0*d)/c*mean_delta_t*nu

def temp(omega_0, delta_omega, m, c, kB):
    return m*c**2.0/(8.0*kB*np.log(2.0))*(delta_omega/omega_0)**2.0

def main():
    DATA_DIR = "../data/data00.csv"
    
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
    
    mean_delta_t = 1/len(time_distances)*np.sum(np.array(time_distances))

    d = 0.1
    c = const.c
    kB = const.k
    mass_Rb_85 = 1.409993199*10**(-25)              #   atomic mass of Rb85 in kg -- https://www.steck.us/alkalidata/rubidium85numbers.pdf
    mass_Rb_87 = 1.443160648*10**(-25)              #   atomic mass of Rb87 in kg -- https://www.steck.us/alkalidata/rubidium87numbers.pdf

    LIST_omega_0 = []
    LIST_delta_omega = []
    
    ### =============== ### 
    ### INITIAL GUESSES ###
    ### =============== ###
    
    #   initial guess and mask for peak 1
    #   =================================
    p0_1 = [-0.1, 0.0121, 0.0003, 1.65]             #   I_0, omega_0, delta_omega, I_y
    t_init_1 = 0.0118 
    t_end_1 = 0.0125
    mask_1 = (time > t_init_1) & (time < t_end_1)
    
    #   initial guess and mask for peak 2
    #   =================================
    p0_2 = [-0.1, 0.0132, 0.0004, 1.60]             #   I_0, omega_0, delta_omega, I_y
    t_init_2 = 0.0126
    t_end_2 = 0.0136
    mask_2 = (time > t_init_2 ) & (time < t_end_2)

    #   initial guess and mask for peak 3
    #   =================================
    p0_3 = [-0.1, 0.01495, 0.0008, 1.54]            #   I_0, omega_0, delta_omega, I_y
    t_init_3 = 0.0148
    t_end_3 = 0.0156
    mask_3 = (time > t_init_3) & (time < t_end_3)
    
    #   initial guess and mask for peak 4
    #   =================================
    p0_4 = [-0.1, 0.0171, 0.0008, 1.48]             #   I_0, omega_0, delta_omega, I_y
    t_init_4 = 0.0168
    t_end_4 = 0.0174
    mask_4 = (time > t_init_4) & (time < t_end_4)
    
    #   initial guess and mask for linear fit
    #   =====================================
    t_init = 0.0137
    U_init = 1.62
    t_end = 0.0193
    U_end = 1.435

    a = (U_end - U_init)/(t_end - t_init)
    U0 = U_end - a*t_end
    p0 = [a, U0]
    mask_all = np.logical_not(mask_1 | mask_2 | mask_3 | mask_4)
    



    ### ==== ###
    ### FITS ###
    ############

    #   fit for peak 1
    #   ==============
    popt_1, pcov_1 = opt.curve_fit(gauss_fit_func, time[mask_1], voltage_3[mask_1], p0_1)
    I_0, omega_0, delta_omega, I_y = popt_1

    print("\n=== PARAMETERS ===")
    print("==================")
    print("I_0, omega_0, delta_omega, I_y = ", popt_1)
    LIST_omega_0.append(popt_1[1])
    LIST_delta_omega.append(popt_1[2])
    
    time_new_1 = np.linspace(t_init_1, t_end_1, 1000)
    voltage_3_fit_peak_1 = gauss_fit_func(time_new_1, *popt_1)
     
    #   fit for peak 2
    #   ==============
    popt_2, pcov_2 = opt.curve_fit(gauss_fit_func, time[mask_2], voltage_3[mask_2], p0_2)
    I_0, omega_0, delta_omega, I_y = popt_2
    
    print("\n=== PARAMETERS ===")
    print("==================")
    print("I_0, omega_0, delta_omega, I_y = ", popt_2)
    LIST_omega_0.append(popt_2[1])
    LIST_delta_omega.append(popt_2[2])
    
    time_new_2 = np.linspace(t_init_2, t_end_2, 1000)
    voltage_3_fit_peak_2 = gauss_fit_func(time_new_2, *popt_2)

    #   fit for peak 3
    #   ==============
    popt_3, pcov_3 = opt.curve_fit(gauss_fit_func, time[mask_3], voltage_3[mask_3], p0_3)
    I_0, omega_0, delta_omega, I_y = popt_3

    print("\n=== PARAMETERS ===")
    print("==================")
    print("I_0, omega_0, delta_omega, I_y = ", popt_3)
    LIST_omega_0.append(popt_3[1])
    LIST_delta_omega.append(popt_3[2])

    time_new_3 = np.linspace(t_init_3, t_end_3, 1000)
    voltage_3_fit_peak_3 = gauss_fit_func(time_new_3, *popt_3)
    
    #   fit for peak 4
    #   ==============
    popt_4, pcov_4 = opt.curve_fit(gauss_fit_func, time[mask_4], voltage_3[mask_4], p0_4)
    I_0, omega_0, delta_omega, I_y = popt_4
    
    print("\n=== PARAMETERS ===")
    print("==================")
    print("I_0, omega_0, delta_omega, I_y = ", popt_4)
    LIST_omega_0.append(popt_4[1])
    LIST_delta_omega.append(popt_4[2])

    time_new_4 = np.linspace(t_init_4, t_end_4, 1000)
    voltage_3_fit_peak_4 = gauss_fit_func(time_new_4, *popt_4)
    
    #   linear fit
    #   ==========
    popt, pcov = opt.curve_fit(lin_fit_func, time[mask_all], voltage_3[mask_all], p0)
    a, U0 = popt
    
    print("\n=== PARAMETERS ===")
    print("===================")
    print("a, U0 = ", popt)
    voltage_3_lin_fit = lin_fit_func(time, *popt)
    



    ### ===== ### 
    ### PLOTS ###
    ### ===== ###

    fig, ax = plt.subplots()
     
    #   plot raw data
    #   =============
    ax.plot(time, voltage_3, color = 'tab:blue')
    #ax.plot(time_to_freq(time, c, d, mean_delta_t)*10**(-9), voltage_3, color = 'tab:blue')                         #   multiply time_to_freq with 10**(-9) to plot in GHz
    
    ax.set_xlabel(r'Zeit $t$ in s')
    #ax.set_xlabel(r'Frequenz $\nu$ in GHz')
    ax.set_ylabel('Spannung $U$ in mV')
    ax.set_ylim(1.405, 1.74)

    ax.grid(True) 

    fig.savefig("../report/figures/plots/PNG/plot-data00-rubidium.png", format = 'png', bbox_inches = 'tight', dpi = 400)
    #fig_i.savefig("../report/figures/plots/EPS/plot-data00-rubidium.eps", format = 'eps', bbox_inches = 'tight')
    fig.savefig("../report/figures/plots/PDF/plot-data00-rubidium.pdf", format = 'pdf', bbox_inches = 'tight')
    #plt.show()
    #tikplotlib.save("../report/figures/tikz/plot-data00-rubidium.tex")
    

    #   plot linear fit
    #   ===============
    ax.plot(time_to_freq(time, c, d, mean_delta_t), voltage_3_lin_fit, color = 'tab:grey')
    #ax.plot(time, voltage_3_lin_fit, color = 'tab:grey')
    
    #   plot fit for peak 1
    #   ===================
    #ax.plot(time_to_freq(time_new_1, c, d, mean_delta_t)*10**(-9), voltage_3_fit_peak_1, color = 'tab:red')         #   multiply time_to_freq with 10**(-9) to plot in GHz
    #ax.vlines(time_to_freq(LIST_omega_0[0], c, d, mean_delta_t)*10**(-9), 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:red', linestyles = 'dashed', linewidth = 1)
    ax.plot(time_new_1, voltage_3_fit_peak_1, color = 'tab:red')         #   multiply time_to_freq with 10**(-9) to plot in GHz
    ax.vlines(LIST_omega_0[0], 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:red', linestyles = 'dashed', linewidth = 1)
    
    #   plot fit for peak 2
    #   ===================
    #ax.plot(time_to_freq(time_new_2, c, d, mean_delta_t)*10**(-9), voltage_3_fit_peak_2, color = 'tab:green')       #   multiply time_to_freq with 10**(-9) to plot in GHz
    #ax.vlines(time_to_freq(LIST_omega_0[1], c, d, mean_delta_t)*10**(-9), 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:green', linestyles = 'dashed', linewidth = 1)
    ax.plot(time_new_2, voltage_3_fit_peak_2, color = 'tab:green')       #   multiply time_to_freq with 10**(-9) to plot in GHz
    ax.vlines(LIST_omega_0[1], 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:green', linestyles = 'dashed', linewidth = 1)

    #   plot fit for peak 3
    #   ===================
    #ax.plot(time_to_freq(time_new_3, c, d, mean_delta_t)*10**(-9), voltage_3_fit_peak_3, color = 'tab:pink')        #   multiply time_to_freq with 10**(-9) to plot in GHz
    #ax.vlines(time_to_freq(LIST_omega_0[2], c, d, mean_delta_t)*10**(-9), 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:pink', linestyles = 'dashed', linewidth = 1)
    ax.plot(time_new_3, voltage_3_fit_peak_3, color = 'tab:pink')        #   multiply time_to_freq with 10**(-9) to plot in GHz
    ax.vlines(LIST_omega_0[2], 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:pink', linestyles = 'dashed', linewidth = 1)

    #   plot fit for peak 4
    #   ===================
    #ax.plot(time_to_freq(time_new_4, c, d, mean_delta_t)*10**(-9), voltage_3_fit_peak_4, color = 'tab:orange')      #   multiply time_to_freq with 10**(-9) to plot in GHz
    #ax.vlines(time_to_freq(LIST_omega_0[3], c, d, mean_delta_t)*10**(-9), 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:orange', linestyles = 'dashed', linewidth = 1)
    ax.plot(time_new_4, voltage_3_fit_peak_4, color = 'tab:orange')      #   multiply time_to_freq with 10**(-9) to plot in GHz
    ax.vlines(LIST_omega_0[3], 0, 1, transform = ax.get_xaxis_transform(), color = 'tab:orange', linestyles = 'dashed', linewidth = 1)


    fig.savefig("../report/figures/plots/PNG/plot-data00-rubidium-fit.png", format = 'png', bbox_inches = 'tight', dpi = 400)
    #fig_i.savefig("../report/figures/plots/EPS/plot-data00-rubidium-fit.eps", format = 'eps', bbox_inches = 'tight')
    fig.savefig("../report/figures/plots/PDF/plot-data00-rubidium-fit.pdf", format = 'pdf', bbox_inches = 'tight')
    #plt.show()
    #tikzplotlib.save("../report/figures/tikz/plot-data00-rubidium-fit.tex")
    
    
    fig, ax = plt.subplots()

    #   plot data without linear fit
    #   ============================
    ax.plot(time, voltage_3 - voltage_3_lin_fit, color = 'tab:green')

    #ax.set_xlim(0.0123, 0.018)
    #ax.set_ylim(-0.18,0.015)
    
    ax.set_xlabel(r'Zeit $t$ in s')
    #ax.set_xlabel(r'Frequenz $\nu$ in GHz')
    ax.set_ylabel('Spannung $U$ in mV')
    ax.grid(True)
    
    fig.savefig("../report/figures/plots/PNG/plot-data00-rubidium-without-lin-fit.png", format = 'png', bbox_inches = 'tight', dpi = 400)
    #fig_i.savefig("../report/figures/plots/EPS/plot-data00-rubidium-without-lin-fit.eps", format = 'eps', bbox_inches = 'tight')
    fig.savefig("../report/figures/plots/PDF/plot-data00-rubidium-without-lin-fit.pdf", format = 'pdf', bbox_inches = 'tight')
    #plt.show()
    #tikzplotlib.save("../report/figures/tikz/plot-data00-rubidium-wihtout-lin-fit.tex")
    




    ### ============================= ###
    ### CALCULATE FREQUENCY DISTANCES ### 
    ### ============================= ###

    print("\n=== PEAK 3, 4 ===")
    print("=================")
    print("freq_dist in GHz = ", time_to_freq((LIST_omega_0[3] - LIST_omega_0[2]), c, d, mean_delta_t)*10**(-9))
    
    print("\n=== PEAK 2, 3 ===")
    print("=================")
    print("freq_dist in GHz = ", time_to_freq((LIST_omega_0[2] - LIST_omega_0[1]), c, d, mean_delta_t)*10**(-9))
    
    print("\n=== PEAK 1, 2 ===")
    print("=================")
    print("freq_dist in GHz = ", time_to_freq((LIST_omega_0[1] - LIST_omega_0[0]), c, d, mean_delta_t)*10**(-9))
    
    print("\n=== PEAK 1, 4 ===")
    print("=================")
    print("freq_dist in GHz = ", time_to_freq((LIST_omega_0[3] - LIST_omega_0[0]), c, d, mean_delta_t)*10**(-9))
    
    '''
    ### ===================== ###
    ### CALCULATE TEMPERATURE ###
    ### ===================== ###
    
    print("\n=== TEMP PEAK 1 ===")
    print("===================")
    
    print("LIST_omega_0[0] - t_init_1 = ", LIST_omega_0[0] - t_init_1)
    print("LIST_delta_omega[0] = ", LIST_delta_omega[0])
    print("(LIST_omega_0[0] - t_init_1)/LIST_delta_omega[0] = ", (LIST_omega_0[0] - t_init_1)/LIST_delta_omega[0])

    #print("time_to_freq(LIST_omega_0[0], c, d, mean_delta_t) = ", time_to_freq(LIST_omega_0[0], c, d, mean_delta_t))
    #print("time_to_freq(LIST_delta_omega[0], c, d, mean_delta_t) = ", time_to_freq(LIST_delta_omega[0], c, d, mean_delta_t))
            
    print("mass_Rb_85 = ", mass_Rb_85)
    print("c = ", c)
    print("kB = ", kB)
    print("temp((LIST_omega_0[0] - t_init_1), LIST_delta_omega[0], mass_Rb_85, c, kB) = ", temp((LIST_omega_0[0] - t_init_1), LIST_delta_omega[0], mass_Rb_85, c, kB))
    
    print("\n=== TEMP PEAK 2 ===")
    print("===================")
    print("omega_0 = ", LIST_omega_0[1])
    print("delta_omega = ", LIST_delta_omega[1])
    print("delta_omega/omega_0 = ", LIST_delta_omega[1]/LIST_omega_0[1])
    print("mass_Rb_85 = ", mass_Rb_85)
    print("c = ", c)
    print("kB = ", kB)
    print("temp in K = ", temp(LIST_omega_0[1], LIST_delta_omega[1], mass_Rb_85, c, kB))
    
    print("\n=== TEMP PEAK 3 ===")
    print("===================")
    print("omega_0 = ", LIST_omega_0[2])
    print("delta_omega = ", LIST_delta_omega[2])
    print("delta_omega/omega_0 = ", LIST_delta_omega[2]/LIST_omega_0[2])
    print("mass_Rb_85 = ", mass_Rb_85)
    print("c = ", c)
    print("kB = ", kB)
    print("temp in K = ", temp(LIST_omega_0[2], LIST_delta_omega[2], mass_Rb_85, c, kB))

    print("\n=== TEMP PEAK 4 ===")
    print("====================")
    print("omega_0 = ", LIST_omega_0[3])
    print("delta_omega = ", LIST_delta_omega[3])
    print("delta_omega/omega_0 = ", LIST_delta_omega[3]/LIST_omega_0[3])
    print("mass_Rb_85 = ", mass_Rb_85)
    print("c = ", c)
    print("kB = ", kB)
    print("temp in K = ", temp(LIST_omega_0[3], LIST_delta_omega[3], mass_Rb_85, c, kB))
    '''

if __name__ == "__main__":
    main()