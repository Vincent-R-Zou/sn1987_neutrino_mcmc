import numpy as np


# 11 events in Kamiokande

Kam_E_valid = np.array([20, 13.5, 7.5, 9.2, 12.8, 35.4, 21.0, 19.8, 8.6, 13.0, 8.9])

Kam_dE_valid = np.array([2.9, 3.2, 2.0, 2.7, 2.9, 8.0, 4.2, 3.2, 2.7, 2.6, 1.9]
)
Kam_t_valid = np.array([0, 0.107, 0.303, 0.324, 0.507, 1.541, 1.728, 1.915, 9.219, 10.433, 12.439])

Kam_c_valid = np.cos(np.array([18, 40, 108, 70, 135, 32, 30, 38, 122, 49, 91])/(180/np.pi))

Kam_dc_valid = np.cos(np.array([18, 27, 32, 30, 23, 16, 18, 22, 30, 26])/(180/np.pi))

Kam_B_valid = np.array([1.6e-5, 1.9e-3, 2.9e-2, 1.2e-2, 2.1e-3, 4.5e-5, 8.2e-5, 1.5e-5, 1.5e-2, 1.9e-3, 1.6e-2])

Kam_B_valid_2009 = np.array([1e-5, 5.4e-4, 3.1e-2, 8.5e-3, 5.3e-4, 5e-6, 1e-5, 1e-5, 1.8e-2, 4e-4, 1.4e-2])


# 5 events usually contributed to background in Kamiokande

Kam_doubt_E = np.array([6.3, 6.5, 5.4, 4.6, 6.5])

Kam_doubt_dE = np.array([1.7, 1.6, 1.4, 1.3, 1.6]
)
Kam_doubt_t = np.array([0.686, 17.641, 20.257, 21.355, 23.814])

Kam_doubt_c = np.cos(np.array([68, 103, 110, 120, 112])/(180/np.pi))

Kam_doubt_dc = np.cos(np.array([77, 39, 50, 50, 50, 50])/(180/np.pi))

Kam_doubt_B = np.array([3.7e-2, 1.6e-2, 3.8e-2, 2.9e-2, 2.8e-2, 3.8e-2])

Kam_doubt_B_2009 = np.array([7.1e-2, 1.4e-2, 7.3e-2, 5.2e-2, 1.8e-2, 7.3e-2])


# events in the first 3.4s in Kamiokande

Kam_t_3s = np.array([0, 0.107, 0.303, 0.324, 0.507, 1.541, 1.728, 1.915])

Kam_E_3s = np.array([20, 13.5, 7.5, 9.2, 12.8, 35.4, 21.0, 19.8])

Kam_dE_3s = np.array([2.9, 3.2, 2.0, 2.7, 2.9, 8.0, 4.2, 3.2])

Kam_c_3s = np.cos(np.array([18, 40, 108, 70, 135, 32, 30, 38])/(180/np.pi))

Kam_dc_3s = np.cos(np.array([18, 27, 32, 30, 23, 16, 18, 22])/(180/np.pi))

Kam_B_3s = np.array([1.6e-5, 1.9e-3, 2.9e-2, 1.2e-2, 2.1e-3, 4.5e-5, 8.2e-5, 1.5e-5])

Kam_B_2009_3s = np.array([1e-5, 5.4e-4, 3.1e-2, 8.5e-3, 5.3e-4, 5e-6, 1e-5, 1e-5])


# Events in Kamiokande

Kam_E = Kam_E_3s

Kam_dE = Kam_dE_3s

Kam_t = Kam_t_3s

Kam_c = Kam_c_3s

Kam_dc = Kam_dc_3s

Kam_B = Kam_B_3s

Kam_B_2009 = Kam_B_2009_3s


# 8 events in IMB

IMB_E_valid = np.array([38, 37, 28, 39, 36, 36, 19, 22])

IMB_dE_valid = np.array([7, 7, 6, 7, 9, 6, 5, 5])

IMB_t_valid = np.array([0, 0.412, 0.650, 1.141, 1.562, 2.684, 5.010, 5.582])

IMB_c_valid = np.cos(np.array([80, 44, 56, 65, 33, 52, 42, 104])/(180/np.pi))

IMB_dc_valid = np.cos(np.array([10, 15, 20, 20, 15, 10, 20, 20])/(180/np.pi))

IMB_B_valid = np.array([0]*len(IMB_E_valid))


# events in the first 3.4s in IMB

IMB_t_3s = np.array([0, 0.412, 0.650, 1.141, 1.562, 2.684])

IMB_E_3s = np.array([38, 37, 28, 39, 36, 36])

IMB_dE_3s = np.array([7, 7, 6, 7, 9, 6])

IMB_c_3s = np.cos(np.array([80, 44, 56, 65, 33, 52])/(180/np.pi))

IMB_dc_3s = np.cos(np.array([10, 15, 20, 20, 15, 10])/(180/np.pi))

IMB_B_3s = np.array([0]*len(IMB_E_3s))


# events in IMB

IMB_E = IMB_E_3s

IMB_dE = IMB_dE_3s

IMB_t = IMB_t_3s

IMB_c = IMB_c_3s

IMB_dc = IMB_dc_3s

IMB_B = IMB_B_3s


# 5 events in Baksan

Baksan_E_valid = np.array([12.0, 18.0, 23.3, 17.0, 20.1])

Baksan_dE_valid = np.array([2.4, 3.6, 4.7, 3.4, 4.0])

Baksan_t_valid = np.array([0, 0.435, 1.710, 7.687, 9.099])

Baksan_c_valid = np.array([0]*len(Baksan_E_valid))

Baksan_dc_valid = np.array([0]*len(Baksan_E_valid))

Baksan_B_valid = np.array([8.4e-4, 1.3e-3, 1.2e-3, 1.3e-3, 1.3e-3])


# events in the first 3.4s in Baksan

Baksan_t_3s = np.array([0, 0.435, 1.710])

Baksan_E_3s = np.array([12.0, 18.0, 23.3])

Baksan_dE_3s = np.array([2.4, 3.6, 4.7])

Baksan_c_3s = np.array([0]*len(Baksan_E_3s))

Baksan_dc_3s = np.array([0]*len(Baksan_E_3s))

Baksan_B_3s = np.array([8.4e-4, 1.3e-3, 1.2e-3])


# events in Baksan

Baksan_E = Baksan_E_3s

Baksan_dE = Baksan_dE_3s

Baksan_t = Baksan_t_3s

Baksan_c = Baksan_c_3s

Baksan_dc = Baksan_dc_3s

Baksan_B = Baksan_B_3s




if __name__ == '__main__':
    print(Kam_E, Kam_dE, Kam_t, Kam_c, Kam_dc, Kam_B)
    print(IMB_E, IMB_dE, IMB_t, IMB_c, IMB_dc, IMB_B)
    print(Baksan_E, Baksan_dE, Baksan_t, Baksan_c, Baksan_dc, Baksan_B)