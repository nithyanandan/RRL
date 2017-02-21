import numpy as NP
from astropy import constants as FCNST
from astropy import units
import scipy.special as SPS
import rrlines as RRL

###############################################################################

def bohr_radius(atomic_number=1, n=1, atomic_mass=1):

    """
    ---------------------------------------------------------------------------
    Calculate Bohr radius for a given atomic number and principal quantum 
    number of electron level

    Inputs:

    atomic_number   [scalar or numpy array] Atomic number of the atom. It is 
                    equal to the number of protons in the nucleus. Must be a 
                    positive integer and greater than or equal to unity. It
                    could be a scalar or a numpy array

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive integer and greater than 
                    or equal to unity (default). Must be a scalar or a numpy 
                    array 

    atomic_mass     [scalar or numpy array] Atomic mass of the atom. It is 
                    equal to the sum of the number of protons and neutrons in 
                    the nucleus. Must be positive and greater than or equal to 
                    unity (default). It must be a scalar or a numpy array

    Inputs atomic_number, n, atomic_mass must have sizes consistent for 
    numpy broadcasting. 

    Output:

    Return Bohr radius for given atomic number and electron level. It is of 
    the same size as input n.
    ---------------------------------------------------------------------------
    """

    try:
        atomic_number, n
    except NameError:
        raise NameError('Inputs atomic number, atomic mass and lower electron level must be specified')

    if not isinstance(atomic_number, (int,NP.ndarray)):
        raise TypeError('Input atomic_number must be a scalar or numpy array')
    atomic_number = NP.asarray(atomic_number).reshape(-1)
    if NP.any(atomic_number < 1):
        raise ValueError('Input atomic_number must be greater than 1')

    if not isinstance(n, (int, NP.ndarray)):
        raise TypeError('Input n must be an integer or numpy array')
    n = NP.asarray(n).reshape(-1)
    if NP.any(n < 1):
        raise ValueError('Lower electron level must be greater than 1')

    if not isinstance(atomic_mass, (int,NP.ndarray)):
        raise TypeError('Input atomic_mass must be a scalar or numpy array')
    atomic_mass = NP.asarray(atomic_mass).reshape(-1)
    if NP.any(atomic_mass < atomic_number):
        raise ValueError('Input atomic_mass must be greater than 1')

    n_protons = atomic_number
    n_neutrons = atomic_mass - atomic_number
    m = n_protons * FCNST.m_p + n_neutrons * FCNST.m_n
    ratio_ryd_m = (1 + FCNST.m_e / m)
    
    return ratio_ryd_m * FCNST.a0 / atomic_number * n**2

###############################################################################

def statistical_weight(atomic_number, n):

    """
    ---------------------------------------------------------------------------
    Calculate statistical weights for given atmoic number and energy level. 
    May not be accurate for non-Hydrogen atoms

    Inputs:

    atomic_number   [scalar] Atomic number of the atom. It is equal to the 
                    number of protons in the nucleus. Must be a positive 
                    integer and greater than or equal to unity. 

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive integer and greater than 
                    or equal to unity. Must be a scalar or same size as 
                    atomic_number

    Output:

    Return statistical weights for energy levels of a given atomic number. It 
    is of the same size as input n.
    ---------------------------------------------------------------------------
    """

    try:
        atomic_number, n
    except NameError:
        raise NameError('Inputs atomic number, atomic mass and lower electron level must be specified')

    if not isinstance(atomic_number, int):
        if isinstance(atomic_number, NP.ndarray):
            if atomic_number.reshape(-1).size != 1:
                raise TypeError('Input atomic_number must be an integer')
            else:
                atomic_number = atomic_number[0]
    if atomic_number < 1:
        raise ValueError('Input atomic_number must be greater than 1')

    if not isinstance(n, (int, NP.ndarray)):
        raise TypeError('Input n must be an integer or numpy array')
    n = NP.asarray(n).reshape(-1)
    if NP.any(n < 1):
        raise ValueError('Lower electron level must be greater than 1')

    if atomic_number == 1:
        g_core = 1
    elif atomic_number == 2:
        g_core = 2
    elif atomic_number == 8:
        g_core = 4
    else:
        raise NotImplementedError('Specified value of input atomic_number not supported')

    return 2 * g_core * n**2

###############################################################################

def oscillator_strength(atomic_number, n, dn=1, transition='absorption'):

    """
    ---------------------------------------------------------------------------
    Calculate oscillator strengths for given atomic number and transition
    between energy levels. 
    Reference: Menzel and Perekis (1935), Burgess (1958), Shaver (1975)
    Verified for Hydrogen (Z=1) using equation (18) in Shaver (1975)

    Inputs:

    atomic_number   [scalar] Atomic number of the atom. It is equal to the 
                    number of protons in the nucleus. Must be a positive 
                    integer and greater than or equal to unity. 

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive integer and greater than 
                    or equal to unity. Must be a scalar or same size as 
                    atomic_number

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity (default). It 
                    must be of the same size as input n

    transition      [string] Specifies a transition denoting 'absorption'
                    (default) or 'emission'.

    Output:

    Return oscillator strengths for a given atomic number and transition 
    between specified energy levels. It is of the same size as input n.
    ---------------------------------------------------------------------------
    """

    try:
        atomic_number, n
    except NameError:
        raise NameError('Inputs atomic number and lower electron level must be specified')

    if not isinstance(atomic_number, (int,NP.ndarray)):
        raise TypeError('Input atomic_number must be a scalar or numpy array')
    atomic_number = NP.asarray(atomic_number).reshape(-1)
    if NP.any(atomic_number < 1):
        raise ValueError('Input atomic_number must be greater than 1')

    if not isinstance(n, (int, NP.ndarray)):
        raise TypeError('Input n must be an integer or numpy array')
    n = NP.asarray(n).reshape(-1)
    if NP.any(n < 1):
        raise ValueError('Lower electron level must be greater than 1')
    
    if not isinstance(dn, (int, NP.ndarray)):
        raise TypeError('Input dn must be an integer or numpy array')
    dn = NP.asarray(dn).reshape(-1)
    if NP.any(dn < 1):
        raise ValueError('Lower electron level must be greater than 1')
    if dn.size != n.size:
        if dn.size != 1:
            raise ValueError('Sizes of inputs n and dn must be same or dn must be a scalar')

    if not isinstance(transition, str):
        raise TypeError('Input transition must be a string')
    if transition.lower() not in ['emission', 'absorption']:
        raise ValueError('Input transition must be set to "absorption" or "emission"')

    # Estimate in "emission" mode where n1 > n2
    n1 = n+dn
    n2 = n
    g2 = statistical_weight(atomic_number, n2)
    f_n1n2_kramers = 64.0 / (3.0 * NP.sqrt(3.0) * NP.pi * g2) / (1.0/n2**2 - 1.0/n1**2)**3 / n1**3 / n2**3

    d = 1 # normalizing factor for transitions between discrete states
    term1 = SPS.hyp2f1(1-n1, -n2, 1, -4*n1*n2/dn**2)
    term2 = SPS.hyp2f1(1-n2, -n1, 1, -4*n1*n2/dn**2)
    delta = term1**2 - term2**2
    if NP.any(NP.isnan(delta)):
        a = 1.0 * n2 / n1
        asymptotic_approx = 1.0 - 0.1728 * (1+a**2) / ((1-a**2) * n2)**(2.0/3.0) - 0.0496 * (1.0 - 4.0/3.0 * a**2 + a**4) / ((1-a**2) * n2)**(4.0/3.0)
        gaunt_n1n2 = d * asymptotic_approx
    else:
        gaunt_n1n2 = NP.sqrt(3.0) * NP.pi * d * (1.0*(n1-n2)/(n1+n2))**(2.0*n1+2.0*n2) * delta * n1 * n2 / (n1-n2)
    f_n1n2 = -1.0 * f_n1n2_kramers * NP.abs(gaunt_n1n2)
    
    if transition.lower() == 'emission':
        return f_n1n2
    else:
        g1 = statistical_weight(atomic_number, n1)
        f_n2n1 = -1.0 * g1 * f_n1n2 / g2
        return f_n2n1

###############################################################################

def einstein_A_coeff(atomic_number, atomic_mass, n, dn=1, screening=False):

    """
    ---------------------------------------------------------------------------
    Calculate the Einstein A-coefficient for spontaneous emission due to 
    transition between two energy levels.
    Reference: Menzel and Perekis (1935)
    Reference: Hilborn (2002)
    Verified for Hydrogen (Z=1) using equation (18) in Shaver (1975)

    Inputs:

    atomic_number   [scalar] Atomic number of the atom. It is equal to the 
                    number of protons in the nucleus. Must be a positive 
                    integer and greater than or equal to unity. 

    atomic_mass     [integer] Atomic mass of the atom. It is equal to the sum
                    of the number of protons and neutrons in the nucleus. Must 
                    be positive and greater than or equal to unity.

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive integer and greater than 
                    or equal to unity. Must be a scalar or same size as 
                    atomic_number

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity (default). It 
                    must be of the same size as input n

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    Return Einstein A-coefficient for spontaneous transition between specified 
    energy levels. It is of the same size as input n.
    ---------------------------------------------------------------------------
    """

    nu = RRL.restframe_freq_recomb(atomic_number, atomic_mass, n, dn=dn, screening=screening)
    f_ul = oscillator_strength(atomic_number, n, dn=dn, transition='emission')
    a_ul = -1.0 * 2 * NP.pi * FCNST.e.to('coulomb')**2 * nu**2 / (FCNST.eps0 * FCNST.m_e * FCNST.c**3) * f_ul # Reference Hilborn (2002)
    return a_ul.decompose()

###############################################################################

def classical_Einstein_A_coeff(atomic_number, atomic_mass, n, dn, screening=False):

    """
    ---------------------------------------------------------------------------
    Calculate approximate Einstein A coefficient for spontaneous transition 
    from an upper to a lower level. It is approximate as it is valid only for
    n >> 1 where some classical approximations can be used.

    Inputs:

    atomic_number   [scalar or numpy array] Atomic number of the atom. It is 
                    equal to the number of protons in the nucleus. Must be 
                    a positive integer and greater than or equal to unity. 
                    Must be a scalar or same size as n

    atomic_mass     [integer] Atomic mass of the atom. It is equal to the sum
                    of the number of protons and neutrons in the nucleus. Must 
                    be positive and greater than or equal to unity.

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive integer and greater than 
                    or equal to unity. Must be a scalar or same size as 
                    atomic_number

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity (default). It 
                    must be of the same size as input n

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    Return approximate Einstein A coeffcient. It is of the same size as inputs 
    atomic_number, n, dn and is an instance of class astropy.units.Quantity
    Results are valid only for n >> 1
    ---------------------------------------------------------------------------
    """

    nu = RRL.restframe_freq_recomb(atomic_number, atomic_mass, n, dn=dn, screening=screening)
    if not isinstance(nu, units.Quantity):
        nu = nu * units.Hertz
    a_lower = bohr_radius(atomic_number, n)
    a_upper = bohr_radius(atomic_number, n+dn)
    power = FCNST.e.to('coulomb')**2 * (a_lower*a_upper) * (2*NP.pi*nu)**4 / (3 * FCNST.c**3) / (4.0*NP.pi*FCNST.eps0) # Larmor's formula for power radiated by a dipole
    einstein_a_coeff = power / (FCNST.h * nu)

    return einstein_a_coeff.decompose()

###############################################################################

def einstein_B_ul_coeff(atomic_number, n, dn=1, atomic_mass=None, screening=False, b_lu=None):

    """
    ---------------------------------------------------------------------------
    Calculate Einstein B coefficient for stimulated transition from an upper 
    to a lower level. It is approximately valid if it is derived from the 
    Einstein A-coefficient which in turn is approximate and valid only for 
    n >> 1. It could also be derived from B_lu which is the Einstein 
    B-coefficient for stimulated transition from lower to upper level in which
    case it is accurate. 

    Inputs:

    atomic_number   [scalar or numpy array] Atomic number of the atom. It is 
                    equal to the number of protons in the nucleus. Must be 
                    a positive integer and greater than or equal to unity 
                    (default=None). Must be a scalar and must be specified

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive integer and greater than 
                    or equal to unity. Must be a scalar or same size as 
                    atomic_number

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity (default). It 
                    must be a scalar or of the same size as input n

    atomic_mass     [integer] Atomic mass of the atom. It is equal to the sum
                    of the number of protons and neutrons in the nucleus. Must 
                    be positive and greater than or equal to unity. 
                    Default=None. This will be used in deriving the approximate 
                    Einstein A-coefficient. If the Einstein B-coefficient b_lu 
                    is specified, this input is ignored.

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity. This will 
                    be used in deriving the approximate Einstein A-coefficient. 
                    It is irrelevant if Einstein B-coefficient b_lu is set.

    b_lu            [scalar or numpy array] Einstein B-coefficient for 
                    stimulated transition from lower to upper level. If 
                    specified then inputs relevant for calculating through
                    Einstein A-coeffcient will be ignored. If not set to 
                    None, it must be a scalar or numpy array (same size as n).
                    It must be specified in units of m^3 / J / s^2

    Output:

    Return Einstein B coeffcient for stimulated emission from upper to lower
    level. It is of the same size as input n and is an instance of 
    class astropy.units.Quantity. Depending on how it was derived, it may be
    valid only for n >> 1. It is in units of m^3 / J / s^2 using the relation
    b_ul = a_ul * c**3 / (8 * pi * h * nu**3) and b_lu = (g_u/g_l) * b_ul
    ---------------------------------------------------------------------------
    """

    if b_lu is not None:
        if not isinstance(b_lu, (int,float,NP.ndarray,units.Quantity)):
            raise TypeError('Input b_lu must be a scalar or numpy array')
        if not isinstance(b_lu, units.Quantity):
            b_lu = NP.asarray(b_lu) * (units.meter)**3 * (units.joule)**-1 * (units.Hertz)**2
        else:
            b_lu = units.Quantity(NP.asarray(b_lu.value).reshape(-1), b_lu.unit)
        g_lower = statistical_weight(atomic_number, n)
        g_upper = statistical_weight(atomic_number, n+dn)
        b_ul = (g_lower / g_upper) * b_lu
    else:
        nu = RRL.restframe_freq_recomb(atomic_number, atomic_mass, n, dn=dn, screening=screening)
        a_ul = einstein_A_coeff(atomic_number, atomic_mass, n, dn, screening=screening)
        b_ul = (FCNST.c)**3 / (8 * NP.pi * FCNST.h * nu**3) * a_ul

    return b_ul

###############################################################################

def einstein_B_lu_coeff(atomic_number, n, dn=1, atomic_mass=None, screening=False, b_ul=None):

    """
    ---------------------------------------------------------------------------
    Calculate Einstein B coefficient for stimulated transition from an lower 
    to upper level. It is approximately valid if it is derived from the 
    Einstein A-coefficient which in turn is approximate and valid only for 
    n >> 1. It could also be derived from b_ul which is the Einstein 
    B-coefficient for stimulated transition from upepr to lower level in which
    case it is accurate. 

    Inputs:

    atomic_number   [scalar or numpy array] Atomic number of the atom. It is 
                    equal to the number of protons in the nucleus. Must be 
                    a positive integer and greater than or equal to unity 
                    (default=None). Must be a scalar and must be specified

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive integer and greater than 
                    or equal to unity. Must be a scalar or same size as 
                    atomic_number

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity (default). It 
                    must be a scalar or of the same size as input n

    atomic_mass     [integer] Atomic mass of the atom. It is equal to the sum
                    of the number of protons and neutrons in the nucleus. Must 
                    be positive and greater than or equal to unity. 
                    Default=None. This will be used in deriving the approximate 
                    Einstein A-coefficient. If the Einstein B-coefficient b_ul 
                    is specified, this input is ignored.

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity. This will 
                    be used in deriving the approximate Einstein A-coefficient. 
                    It is irrelevant if Einstein B-coefficient b_ul is set.

    b_ul            [scalar or numpy array] Einstein B-coefficient for 
                    stimulated transition from lower to upper level. If 
                    specified then inputs relevant for calculating through
                    Einstein A-coeffcient will be ignored. If not set to 
                    None, it must be a scalar or numpy array (same size as n).
                    It must be specified in units of m^3 / J / s^2

    Output:

    Return Einstein B coeffcient for stimulated emission from upper to lower
    level. It is of the same size as input n and is an instance of 
    class astropy.units.Quantity. Depending on how it was derived, it may be
    valid only for n >> 1. It is in units of m^3 / J / s^2 using the relation
    b_ul = a_ul * c**3 / (8 * pi * h * nu**3) and b_lu = (g_u/g_l) * b_ul
    ---------------------------------------------------------------------------
    """

    if b_ul is not None:
        if not isinstance(b_ul, (int,float,NP.ndarray,units.Quantity)):
            raise TypeError('Input b_ul must be a scalar or numpy array')
        if not isinstance(b_ul, units.Quantity):
            b_ul = NP.asarray(b_ul) * (units.meter)**3 * (units.joule)**-1 * (units.Hertz)**2
        else:
            b_ul = units.Quantity(NP.asarray(b_ul.value).reshape(-1), b_ul.unit)
    else:
        nu = RRL.restframe_freq_recomb(atomic_number, atomic_mass, n, dn=dn, screening=screening)
        a_ul = einstein_A_coeff(atomic_number, atomic_mass, n, dn, screening=screening)
        b_ul = (FCNST.c)**3 / (8 * NP.pi * FCNST.h * nu**3) * a_ul
    g_lower = statistical_weight(atomic_number, n)
    g_upper = statistical_weight(atomic_number, n+dn)
    b_lu = (g_upper / g_lower) * b_ul

    return b_lu

###############################################################################

def doppler_broadened_rrline_profile(mass_particle, temperature, nu_0, nu,
                                     rms_turbulent_velocity=0.0):

    """
    ---------------------------------------------------------------------------
    Estimate Doppler broadened recombination line profiles
    # Reference: Rybicki & Lightman (Section 10.6)

    Inputs:

    mass_particle   [scalar or numpy array] Mass of particle (kg). If specified
                    as numpy array, it must be of same size as input nu. 
                    Could also be specified as an instance of class 
                    astropy.units.Quantity 

    temperature     [scalar or numpy array] Temperature (K). If specified
                    as numpy array, it must be of same size as input nu.
                    Could also be specified as an instance of class 
                    astropy.units.Quantity 

    nu_0            [scalar or numpy array] Line-center frequency (in Hz). 
                    Should be of same size as input nu. Could also be 
                    specified as an instance of class astropy.units.Quantity 

    nu              [scalar or numpy array] Frequency (Hz) at which line 
                    profile is to be estimated. If specified as numpy array, 
                    it must be of same size as input nu. Could also be 
                    specified as an instance of class astropy.units.Quantity

    rms_turbulent_velocity
                    [scalar or numpy array] RMS of turbulent velocity (in 
                    km/s). If specified as numpy array, it must be of same 
                    size as input nu. Could also be specified as an instance 
                    of class astropy.units.Quantity Quantity

    Output:

    Normalized Doppler-broadened line profile. Same size as input nu_0. It will
    be returned as an instance of class astropy.units.Quantity. It will have 
    units of 'second'
    ---------------------------------------------------------------------------
    """

    try:
        mass_particle, temperature, nu_0, nu
    except NameError:
        raise NameError('Inputs mass_particle, temperature, nu_0 and nu must be specified')

    if not isinstance(mass_particle, (int,float,NP.ndarray,units.Quantity)):
        raise TypeError('Input mass_particle must be a scalar or a numpy array')
    if not isinstance(mass_particle, units.Quantity):
        mass_particle = NP.asarray(mass_particle).reshape(-1) * units.kilogram
    else:
        mass_particle = units.Quantity(NP.asarray(mass_particle.value).reshape(-1), mass_particle.unit)
    if NP.any(mass_particle <= 0.0*units.kilogram):
        raise ValueError('Input mass_particle must be positive')

    if not isinstance(temperature, (int,float,NP.ndarray,units.Quantity)):
        raise TypeError('Input temperature must be a scalar or a numpy array')
    if not isinstance(temperature, units.Quantity):
        temperature = NP.asarray(temperature).reshape(-1) * units.Kelvin
    else:
        temperature = units.Quantity(NP.asarray(temperature.value).reshape(-1), temperature.unit)
    if NP.any(temperature <= 0.0*units.Kelvin):
        raise ValueError('Input temperature must be positive')

    if not isinstance(nu, (int,float,NP.ndarray,units.Quantity)):
        raise TypeError('Input nu must be a scalar or a numpy array')
    if not isinstance(nu, units.Quantity):
        nu = NP.asarray(nu).reshape(-1) * units.Hertz
    else:
        nu = units.Quantity(NP.asarray(nu.value).reshape(-1), nu.unit)
    if NP.any(nu <= 0.0*units.Hertz):
        raise ValueError('Input nu must be positive')

    if not isinstance(nu_0, (int,float,NP.ndarray,units.Quantity)):
        raise TypeError('Input nu_0 must be a scalar or a numpy array')
    if not isinstance(nu_0, units.Quantity):
        nu_0 = NP.asarray(nu_0).reshape(-1) * units.Hertz
    else:
        nu_0 = units.Quantity(NP.asarray(nu_0.value).reshape(-1), nu_0.unit)
    if NP.any(nu_0 <= 0.0*units.Hertz):
        raise ValueError('Input nu_0 must be positive')
    
    if not isinstance(rms_turbulent_velocity, (int,float,NP.ndarray,units.Quantity)):
        raise TypeError('Input rms_turbulent_velocity must be a scalar or a numpy array')
    if not isinstance(rms_turbulent_velocity, units.Quantity):
        rms_turbulent_velocity = NP.asarray(rms_turbulent_velocity).reshape(-1) * units.kilometer / units.second
    else:
        rms_turbulent_velocity = units.Quantity(NP.asarray(rms_turbulent_velocity.value).reshape(-1), rms_turbulent_velocity.unit)
    if NP.any(rms_turbulent_velocity < 0.0 * units.kilometer / units.second):
        raise ValueError('Input rms_turbulent_velocity must not be negative')
    
    if mass_particle.size != nu.size:
        if mass_particle.size != 1:
            raise ValueError('Input mass_particle must contain one or same number of elements as input nu')

    if temperature.size != nu.size:
        if temperature.size != 1:
            raise ValueError('Input temperature must contain one or same number of elements as input nu')

    if nu_0.size != nu.size:
        if nu_0.size != 1:
            raise ValueError('Input nu_0 must contain one or same number of elements as input nu')

    if rms_turbulent_velocity.size != nu.size:
        if rms_turbulent_velocity.size != 1:
            raise ValueError('Input rms_turbulent_velocity must contain one or same number of elements as input nu')

    dnu = nu_0 / FCNST.c * NP.sqrt(2 * FCNST.k_B * temperature / mass_particle + rms_turbulent_velocity**2) 

    line_profile = 1.0 / (NP.sqrt(NP.pi) * dnu) * NP.exp(-(nu-nu_0)**2 / dnu**2)

    return line_profile

###############################################################################

def saha_boltzmann_equation(atomic_number, n, T_e, N_e, N_ion=None,
                            atomic_mass=None, departure_coeff=1.0,
                            screening=False):

    """
    ---------------------------------------------------------------------------
    Saha-Boltzmann equation to determine the number density of atoms with 
    electrons populating the specified energy level
    # Reference: Shaver (1975) equation (2)

    Inputs:

    atomic_number   [scalar or numpy array] Atomic number of the atom. It is 
                    equal to the number of protons in the nucleus. Must be 
                    positive and greater than or equal to unity. It can be 
                    specified as a scalar or a numpy array with size equal to 
                    that of input n

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive and greater than or equal 
                    to unity unity.

    T_e             [scalar or numpy array] Electron temperature (in K). Must be 
                    positive. It can be specified as a scalar or a numpy array 
                    with size equal to that of input n

    N_e             [scalar or numpy array] Number density (in cm^-3) of 
                    electrons. Must be positive. It can be specified as a 
                    scalar or a numpy array with size equal to that of input n

    N_ion           [scalar or numpy array] Number density (in cm^-3) of 
                    ions. Must be positive. It can be specified as a 
                    scalar or a numpy array with size equal to that of input n.
                    If set to None, it will be assumed to be equal to N_e

    atomic_mass     [scalar or numpy array] Atomic mass of the atom. It is 
                    equal to the sum of the number of protons and neutrons 
                    in the nucleus. Must be positive and greater than or equal 
                    to unity. It can be specified as a scalar or a numpy array 
                    with size equal to that of input n

    departure_coeff [scalar or numpy array] Coefficients signifying departures
                    of populations from LTE values. It must lie between 0 and
                    1. Usually, for large n, these departure coefficients tend
                    towards unity signifying thermalization and LTE. It can be 
                    a scalar or a numpy array of size equal to that of input n. 

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    Number density (cm^-3) of atoms with electrons in the specified energy 
    level
    ---------------------------------------------------------------------------
    """

    try:
        atomic_number, n, T_e, N_e
    except NameError:
        raise NameError('Inputs atomic_number, energy level n, electron temperature T_e, and electron number density N_e must be specified')

    if not isinstance(T_e, (int,float,NP.ndarray,units.Quantity)):
        raise TypeError('Input T_e must be a scalar or a numpy array')
    if not isinstance(T_e, units.Quantity):
        T_e = NP.asarray(T_e).reshape(-1) * units.Kelvin
    else:
        T_e = units.Quantity(NP.asarray(T_e.value).reshape(-1), T_e.unit)
    if NP.any(T_e <= 0.0*units.Kelvin):
        raise ValueError('Input T_e must be positive')

    if not isinstance(N_e, (int,float,NP.ndarray,units.Quantity)):
        raise TypeError('Input N_e must be a scalar or a numpy array')
    if not isinstance(N_e, units.Quantity):
        N_e = NP.asarray(N_e).reshape(-1) / units.cm**3
    else:
        N_e = units.Quantity(NP.asarray(N_e.value).reshape(-1), N_e.unit)
    if NP.any(N_e <= 0.0/units.cm**3):
        raise ValueError('Input N_e must be positive')

    if not isinstance(departure_coeff, (int,float,NP.ndarray)):
        raise TypeError('Input departure_coeff must be a scalar or a numpy array')
    departure_coeff = NP.asarray(departure_coeff).reshape(-1)
    if NP.any(NP.logical_or(departure_coeff < 0.0, departure_coeff > 1.0)):
        raise ValueError('Input departure_coeff must lie in the range [0,1]')
    if departure_coeff.size != n.size:
        if departure_coeff.size != 1:
            raise ValueError('Departure coeff must be a scalar or a numpy array with same size as input n')
    
    if N_ion is None:
        N_ion = N_e
    if not isinstance(N_ion, (int,float,NP.ndarray,units.Quantity)):
        raise TypeError('Input N_ion must be a scalar or a numpy array')
    if not isinstance(N_ion, units.Quantity):
        N_ion = NP.asarray(N_ion).reshape(-1) / units.cm**3
    else:
        N_ion = units.Quantity(NP.asarray(N_ion.value).reshape(-1), N_ion.unit)
    if NP.any(N_ion <= 0.0/units.cm**3):
        raise ValueError('Input N_ion must be positive')
    
    nu_n = RRL.restframe_freq_recomb(atomic_number, atomic_mass, n, dn=NP.asarray(NP.inf).reshape(-1), screening=screening)
    chi_n = FCNST.h * nu_n / (FCNST.k_B * T_e)
    g_n = statistical_weight(atomic_number, n)
    N_n = departure_coeff * N_e * N_ion * (FCNST.h**2 / (2.0*NP.pi*FCNST.m_e*FCNST.k_B*T_e))**1.5 * (0.5*g_n) * NP.exp(chi_n)

    return N_n.to('cm^-3')

###############################################################################

def number_density_with_energy_level(atomic_number, n, T_e, N_e, N_ion=None,
                                     atomic_mass=None, departure_coeff=1.0,
                                     screening=False):

    """
    ---------------------------------------------------------------------------
    Alias for the Saha-Boltzmann equation to determine the number density of 
    atoms with electrons populating the specified energy level
    # Reference: Shaver (1975) equation (2)

    Inputs:

    atomic_number   [scalar or numpy array] Atomic number of the atom. It is 
                    equal to the number of protons in the nucleus. Must be 
                    positive and greater than or equal to unity. It can be 
                    specified as a scalar or a numpy array with size equal to 
                    that of input n

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive and greater than or equal 
                    to unity unity.

    T_e             [scalar or numpy array] Electron temperature (in K). Must be 
                    positive. It can be specified as a scalar or a numpy array 
                    with size equal to that of input n

    N_e             [scalar or numpy array] Number density (in cm^-3) of 
                    electrons. Must be positive. It can be specified as a 
                    scalar or a numpy array with size equal to that of input n

    N_ion           [scalar or numpy array] Number density (in cm^-3) of 
                    ions. Must be positive. It can be specified as a 
                    scalar or a numpy array with size equal to that of input n.
                    If set to None, it will be assumed to be equal to N_e

    atomic_mass     [scalar or numpy array] Atomic mass of the atom. It is 
                    equal to the sum of the number of protons and neutrons 
                    in the nucleus. Must be positive and greater than or equal 
                    to unity. It can be specified as a scalar or a numpy array 
                    with size equal to that of input n

    departure_coeff [scalar or numpy array] Coefficients signifying departures
                    of populations from LTE values. It must lie between 0 and
                    1. Usually, for large n, these departure coefficients tend
                    towards unity signifying thermalization and LTE. It can be 
                    a scalar or a numpy array of size equal to that of input n. 

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    Number density (cm^-3) of atoms with electrons in the specified energy 
    level
    ---------------------------------------------------------------------------
    """

    return saha_boltzmann_equation(atomic_number, n, T_e, N_e, N_ion=N_ion, atomic_mass=atomic_mass, departure_coeff=departure_coeff, screening=screening)

###############################################################################

def line_absorption_coefficient(atomic_number, atomic_mass, n, dn, nu, T_e, N_e, N_ion=None, N_l=None, N_u=None, nu_0=None, lte=True, rms_turbulent_velocity=0.0, screening=False):

    """
    ---------------------------------------------------------------------------
    Estimate line absorption coefficient (also known as line opacity 
    coefficient) under Local Thermodynamic Equilibrium (LTE) or otherwise. 
    Note: I use a slightly different convention than the standard one. My 
    definition differs from that in Radiative Processes by Rybicki & Lightman 
    by a factor c / (4 * pi).

    Inputs:

    atomic_number   [scalar or numpy array] Atomic number of the atom. It is 
                    equal to the number of protons in the nucleus. Must be 
                    positive and greater than or equal to unity. It can be 
                    specified as a scalar or a numpy array with size equal to 
                    that of input n

    atomic_mass     [scalar or numpy array] Atomic mass of the atom. It is 
                    equal to the sum of the number of protons and neutrons 
                    in the nucleus. Must be positive and greater than or equal 
                    to unity. It can be specified as a scalar or a numpy array 
                    with size equal to that of input n

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive and greater than or equal 
                    to unity unity.

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity (default). It 
                    must be a scalar or of the same size as input n

    nu              [scalar or numpy array] Frequency (Hz) at which line 
                    profile is to be estimated. If specified as numpy array, 
                    it must be of same size as input nu. Could also be 
                    specified as an instance of class astropy.units.Quantity

    T_e             [scalar or numpy array] Electron temperature (in K). Must be 
                    positive. It can be specified as a scalar or a numpy array 
                    with size equal to that of input n

    N_e             [scalar or numpy array] Number density (in cm^-3) of 
                    electrons. Must be positive. It can be specified as a 
                    scalar or a numpy array with size equal to that of input n

    N_ion           [scalar or numpy array] Number density (in cm^-3) of 
                    ions. Must be positive. It can be specified as a 
                    scalar or a numpy array with size equal to that of input n.
                    If set to None (default), it will be assumed to be equal 
                    to N_e

    N_l             [scalar or numpy array] Number density (in cm^-3) of
                    particles with electrons occupying the energy level n. It
                    can be a scalar or numpy array with size equal to that of
                    input n. If input lte is set to False, this must be provided
                    in order to estimate under non-equilibrium conditions. If
                    input lte is set to True, this value is ignored and 
                    computed from the Saha-Boltzmann equation which is valid
                    under LTE. 

    N_u             [scalar or numpy array] Number density (in cm^-3) of
                    particles with electrons occupying the energy level n+dn. 
                    It can be a scalar or numpy array with size equal to that 
                    of input n. If input lte is set to False, this must be 
                    provided in order to estimate under non-equilibrium 
                    conditions. If input lte is set to True, this value is 
                    ignored and computed inherently from the Saha-Boltzmann 
                    equation which is valid under LTE. 

    nu_0            [scalar or numpy array] Line-center frequency (in Hz). 
                    Should be of same size as input nu. Could also be 
                    specified as an instance of class astropy.units.Quantity.
                    If set to None, it will be assumed to be equal to the 
                    input nu in which case the line-center opacity coefficients
                    are estimated. 

    rms_turbulent_velocity
                    [scalar or numpy array] RMS of turbulent velocity (in 
                    km/s). If specified as numpy array, it must be of same 
                    size as input nu. Could also be specified as an instance 
                    of class astropy.units.Quantity Quantity. Default=0.

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    Line opacity coefficients (also known as line absorption coefficients) in 
    units of 1/m estimated for the specified energy levels, frequencies and
    physical conditions under LTE or otherwise. It is of same size as inputs
    nu and/or n
    ---------------------------------------------------------------------------
    """

    try:
        atomic_number, atomic_mass
    except NameError:
        raise NameError('Inputs atomic_number and atomic_mass must be specified')

    if not isinstance(atomic_number, int):
        raise TypeError('Input atomic_number must be an integer')
    if atomic_number < 1:
        raise ValueError('Input atomic_number must be greater than 1')

    if not isinstance(atomic_mass, int):
        raise TypeError('Input atomic_mass must be an integer')
    if atomic_mass < atomic_number:
        raise ValueError('Inout atomic_mass must be greater than or equal to atomic number')

    if not isinstance(n, (int, NP.ndarray)):
        raise TypeError('Input n must be an integer or numpy array')
    n = NP.asarray(n).reshape(-1)
    if NP.any(n < 1):
        raise ValueError('Lower electron level must be greater than 1')

    if not isinstance(dn, (int, NP.ndarray)):
        if not NP.isinf(dn):
            raise TypeError('Input dn must be an integer or numpy array')
    dn = NP.asarray(dn).reshape(-1)
    if NP.any(dn < 1):
        raise ValueError('Lower electron level must be greater than 1')
    if dn.size != n.size:
        if dn.size != 1:
            raise ValueError('Sizes of inputs n and dn must be same, else dn must be a scalar')

    if nu_0 is None:
        nu_0 = nu

    if not isinstance(lte, bool):
        raise TypeError('Input lte must be boolean')

    a_ul = einstein_A_coeff(atomic_number, atomic_mass, n, dn=dn, screening=screening)
    g_u = statistical_weight(atomic_number, n+dn)
    g_l = statistical_weight(atomic_number, n)

    if not lte:
        if (N_l is None) or (N_u is None):
            raise ValueError('When LTE is not valid, both inputs N_l and N_u must be specified')

        if not isinstance(N_l, (int,float,NP.ndarray,units.Quantity)):
            raise TypeError('Input N_l must be a scalar or numpy array')
        if not isinstance(N_l, units.Quantity):
            N_l = NP.asarray(N_l).reshape(-1) * units.cm**-3
        else:
            N_l = units.Quantity(NP.asarray(N_l.value).reshape(-1), N_l.unit)
        if NP.any(N_l < 0.0 * units.cm**-3):
            raise ValueError('Input N_l must not be negative')

        if not isinstance(N_u, (int,float,NP.ndarray,units.Quantity)):
            raise TypeError('Input N_u must be a scalar or numpy array')
        if not isinstance(N_u, units.Quantity):
            N_u = NP.asarray(N_u).reshape(-1) * units.cm**-3
        else:
            N_u = units.Quantity(NP.asarray(N_u.value).reshape(-1), N_u.unit)
        if NP.any(N_u < 0.0 * units.cm**-3):
            raise ValueError('Input N_u must not be negative')

        expr_in_parenthesis = 1.0 - (g_l*N_u)/(g_u*N_l)
    else:
        N_l = number_density_with_energy_level(atomic_number, n, T_e, N_e, N_ion=N_ion, atomic_mass=atomic_mass, screening=screening)
        expr_in_parenthesis = 1.0 - NP.exp(-1.0 * FCNST.h * nu / FCNST.k_B / T_e)

    n_protons = atomic_number
    n_neutrons = atomic_mass - atomic_number
    m_nucleus = n_protons * FCNST.m_p + n_neutrons * FCNST.m_n
    line_profile = doppler_broadened_rrline_profile(m_nucleus, T_e, nu_0, nu, rms_turbulent_velocity=rms_turbulent_velocity)

    absorption_coeff = FCNST.c**2 / (8.0 * NP.pi * nu**2) * N_l * (g_u/g_l) * a_ul * expr_in_parenthesis * line_profile

    return absorption_coeff.decompose()

###############################################################################

def line_opacity_coefficient(atomic_number, atomic_mass, n, dn, nu, T_e, N_e, N_ion=None, N_l=None, N_u=None, nu_0=None, lte=True, rms_turbulent_velocity=0.0, screening=False):

    """
    ---------------------------------------------------------------------------
    An alias for the function line_absoprtion_coefficient()
    Estimate line opacity coefficient (also known as line absorption 
    coefficient) under Local Thermodynamic Equilibrium (LTE) or otherwise. 
    Note: I use a slightly different convention than the standard one. My 
    definition differs from that in Radiative Processes by Rybicki & Lightman 
    by a factor c / (4 * pi).

    Inputs:

    atomic_number   [scalar or numpy array] Atomic number of the atom. It is 
                    equal to the number of protons in the nucleus. Must be 
                    positive and greater than or equal to unity. It can be 
                    specified as a scalar or a numpy array with size equal to 
                    that of input n

    atomic_mass     [scalar or numpy array] Atomic mass of the atom. It is 
                    equal to the sum of the number of protons and neutrons 
                    in the nucleus. Must be positive and greater than or equal 
                    to unity. It can be specified as a scalar or a numpy array 
                    with size equal to that of input n

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive and greater than or equal 
                    to unity unity.

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity (default). It 
                    must be a scalar or of the same size as input n

    nu              [scalar or numpy array] Frequency (Hz) at which line 
                    profile is to be estimated. If specified as numpy array, 
                    it must be of same size as input nu. Could also be 
                    specified as an instance of class astropy.units.Quantity

    T_e             [scalar or numpy array] Electron temperature (in K). Must be 
                    positive. It can be specified as a scalar or a numpy array 
                    with size equal to that of input n

    N_e             [scalar or numpy array] Number density (in cm^-3) of 
                    electrons. Must be positive. It can be specified as a 
                    scalar or a numpy array with size equal to that of input n

    N_ion           [scalar or numpy array] Number density (in cm^-3) of 
                    ions. Must be positive. It can be specified as a 
                    scalar or a numpy array with size equal to that of input n.
                    If set to None (default), it will be assumed to be equal 
                    to N_e

    N_l             [scalar or numpy array] Number density (in cm^-3) of
                    particles with electrons occupying the energy level n. It
                    can be a scalar or numpy array with size equal to that of
                    input n. If input lte is set to False, this must be provided
                    in order to estimate under non-equilibrium conditions. If
                    input lte is set to True, this value is ignored and 
                    computed from the Saha-Boltzmann equation which is valid
                    under LTE. 

    N_u             [scalar or numpy array] Number density (in cm^-3) of
                    particles with electrons occupying the energy level n+dn. 
                    It can be a scalar or numpy array with size equal to that 
                    of input n. If input lte is set to False, this must be 
                    provided in order to estimate under non-equilibrium 
                    conditions. If input lte is set to True, this value is 
                    ignored and computed inherently from the Saha-Boltzmann 
                    equation which is valid under LTE. 

    nu_0            [scalar or numpy array] Line-center frequency (in Hz). 
                    Should be of same size as input nu. Could also be 
                    specified as an instance of class astropy.units.Quantity.
                    If set to None, it will be assumed to be equal to the 
                    input nu in which case the line-center opacity coefficients
                    are estimated. 

    rms_turbulent_velocity
                    [scalar or numpy array] RMS of turbulent velocity (in 
                    km/s). If specified as numpy array, it must be of same 
                    size as input nu. Could also be specified as an instance 
                    of class astropy.units.Quantity Quantity. Default=0.

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    Line opacity coefficients (also known as line absorption coefficients) in 
    units of 1/m estimated for the specified energy levels, frequencies and
    physical conditions under LTE or otherwise. It is of same size as inputs
    nu and/or n
    ---------------------------------------------------------------------------
    """

    return line_absorption_coefficient(atomic_number, atomic_mass, n, dn, nu, T_e, N_e, N_ion=N_ion, N_l=N_l, N_u=N_u, nu_0=nu_0, lte=lte, rms_turbulent_velocity=rms_turbulent_velocity, screening=screening)

###############################################################################
