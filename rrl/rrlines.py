import numpy as NP
from astropy import constants as FCNST
from astropy import units

def restframe_freq_recomb(atomic_number, atomic_mass, n, dn=1, screening=False):

    """
    ---------------------------------------------------------------------------
    Determine rest frame frequencies of photons emitted from electronic
    transitions for an atom of specific atomic and mass numbers, and the
    principal quantum number levels of electronic transitions

    Inputs:

    atomic_number   [integer] Atomic number of the atom. It is equal to the
                    number of protons in the nucleus. Must be positive and
                    greater than or equal to unity.

    atomic_mass     [integer] Atomic mass of the atom. It is equal to the sum
                    of the number of protons and neutrons in the nucleus. Must 
                    be positive and greater than or equal to unity.

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive and greater than or 
                    equal to unity.

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity (default). It 
                    must be of the same size as input n

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    The rest frame frequency (Hz) for the specified electronic transitions. It
    is of the same size as inputs n and dn
    ---------------------------------------------------------------------------
    """

    try:
        atomic_number, atomic_mass, n
    except NameError:
        raise NameError('Inputs atomic_number, atomic_mass and lower electron level, n, must be specified')

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

    if not isinstance(screening, bool):
        raise TypeError('Input screening must be a boolean')

    n_protons = atomic_number
    n_neutrons = atomic_mass - atomic_number
    if screening:
        z_eff = 1
    else:
        z_eff = atomic_number
    m = n_protons * FCNST.m_p + n_neutrons * FCNST.m_n
    ryd_m = FCNST.Ryd / (1 + FCNST.m_e / m)

    nu = z_eff**2 * ryd_m * FCNST.c * ((1.0/n)**2 - (1.0/(n+dn))**2)

    return nu

###############################################################################
    
def redshifted_freq_recomb(atomic_number, atomic_mass, n, dn=1, z=0.0,
                           screening=False):

    """
    ---------------------------------------------------------------------------
    Determine redshifted frequencies of photons emitted from electronic
    transitions for an atom of specific atomic and mass numbers, and the
    principal quantum number levels of electronic transitions

    Inputs:

    atomic_number   [integer] Atomic number of the atom. It is equal to the
                    number of protons in the nucleus. Must be positive and
                    greater than or equal to unity.

    atomic_mass     [integer] Atomic mass of the atom. It is equal to the sum
                    of the number of protons and neutrons in the nucleus. Must 
                    be positive and greater than or equal to unity.

    n               [scalar or numpy array] Principal quantum number of 
                    electron orbits. Must be positive and greater than or equal 
                    to unity unity.

    dn              [scalar or numpy array] Difference in principal quantum
                    number making the transition from n+dn --> n. It must be
                    positive and greater than or equal to unity. It must be of 
                    the same size as input n.

    z               [scalar or numpy array] The redshift (when positive) or 
                    blueshift (when negative) by which the recombination lines
                    are shifted. Default=0

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    The rest frame frequency (Hz) for the specified electronic transitions. It
    is of the shape size(n) x size(z)
    ---------------------------------------------------------------------------
    """

    nu = restframe_freq_recomb(atomic_number, atomic_mass, n, dn=dn, screening=screening)
    if not isinstance(z, (int, float, NP.ndarray)):
        raise TypeError('Input must be a scalar or a numpy array')
    else:
        z = NP.asarray(z).reshape(1,-1)
        nu = nu.reshape(-1,1) / (1.0 + z)
        return nu

###############################################################################

def check_recomb_transitions_in_freq_range(freq_min, freq_max, atomic_number, atomic_mass, n, dn, z=0.0, screening=False):

    """
    ---------------------------------------------------------------------------
    Check if the recomination lines from specified electronic transitions are
    in a given frequency range

    Inputs:

    freq_min        [scalar] Minimum in the frequency range (Hz)

    freq_max        [scalar] Maximum in the frequency range (Hz)

    atomic_number   [integer] Atomic number of the atom. It is equal to the
                    number of protons in the nucleus. Must be positive and
                    greater than or equal to unity.

    atomic_mass     [integer] Atomic mass of the atom. It is equal to the sum
                    of the number of protons and neutrons in the nucleus. Must 
                    be positive and greater than or equal to unity.

    n               [scalar] Principal quantum number of lower electron orbit. 
                    Must be positive and greater than or equal to unity unity.

    dn              [scalar] Difference in principal quantum number making the 
                    transition from n+dn --> n. It must be positive and greater 
                    than or equal to unity. 

    z               [scalar or numpy array] The redshift (when positive) or 
                    blueshift (when negative) by which the recombination lines
                    are shifted. Default=0

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    Output:

    -1 if the recombination line is below the frequency range, 0 if it is in 
    the frequency range and +1 if it is above the frequency range
    ---------------------------------------------------------------------------
    """

    try:
        freq_min, freq_max, atomic_number, atomic_mass, n, dn
    except NameError:
        raise NameError('Inputs freq_min, freq_max, atomic_number, atomic_mass, n, dn must be specified')

    if not isinstance(freq_min, (int,float,units.Quantity)):
        raise TypeError('Input freq_min must be a scalar')
    if not isinstance(freq_min, units.Quantity):
        freq_min = freq_min * units.Hertz
    if freq_min <= 0.0 * units.Hertz:
        raise ValueError('Input freq_min must be positive')

    if not isinstance(freq_max, (int,float,units.Quantity)):
        raise TypeError('Input freq_max must be a scalar')
    if not isinstance(freq_max, units.Quantity):
        freq_max = freq_max * units.Hertz
    if freq_max <= freq_min:
        raise ValueError('Input freq_max must be greater than freq_min')

    if not isinstance(n, int):
        raise TypeError('Input n must be a scalar')
    if n < 1:
        raise ValueError('Input n must be greater than 1')

    if not isinstance(dn, int):
        raise TypeError('Input dn must be a scalar')
    if dn < 1:
        raise ValueError('Input dn must be greater than 1')

    if not isinstance(z, (int, float)):
        raise TypeError('Input z must be a scalar')
    
    nu_z = redshifted_freq_recomb(atomic_number, atomic_mass, n, dn=dn, z=z, screening=screening)
    if nu_z.size > 1:
        raise ValueError('Returned value must contain only one element')
    nu_z = nu_z[0]
    if nu_z > freq_max:
        return 1
    elif nu_z < freq_min:
        return -1
    else:
        return 0
    
###############################################################################

def search_transitions_in_freq_range(freq_min, freq_max, atomic_number,
                                     atomic_mass, n_min=1, n_max=1000,
                                     dn_min=1, dn_max=10, z=0.0,
                                     screening=False, extendsearch=None):

    """
    ---------------------------------------------------------------------------
    Search for electronic transitions of recombination lines at a specified 
    redshift that lie within the specified frequency range

    Inputs:

    freq_min        [scalar] Minimum in the frequency range (Hz)

    freq_max        [scalar] Maximum in the frequency range (Hz)

    atomic_number   [integer] Atomic number of the atom. It is equal to the
                    number of protons in the nucleus. Must be positive and
                    greater than or equal to unity.

    atomic_mass     [integer] Atomic mass of the atom. It is equal to the sum
                    of the number of protons and neutrons in the nucleus. Must 
                    be positive and greater than or equal to unity.

    n_min           [scalar] Minimum in the range of principal quantum numbers 
                    of lower electron orbit to search for transitions.
                    Must be positive and greater than or equal to unity unity.

    n_max           [scalar] Maximum in the range of principal quantum numbers 
                    of lower electron orbit to search for transitions.
                    Must be positive and greater than or equal to unity unity.

    dn_min          [scalar] Minimum in the range of difference in principal 
                    quantum numbers search for transitions. Must be positive 
                    and greater than or equal to unity unity.

    dn_max          [scalar] Maximum in the range of difference in principal 
                    quantum numbers search for transitions. Must be positive 
                    and greater than or equal to unity unity.

    z               [scalar or numpy array] The redshift (when positive) or 
                    blueshift (when negative) by which the recombination lines
                    are shifted. Default=0

    screening       [boolean] If set to False (default), assume the effective
                    charge is equal to the number of protons. If set to True,
                    assume the charges from the nucleus are screened and the
                    effecctive nuclear charge is equal to unity.

    extendsearch    [None or dictionary] Specifies if the search should be 
                    extended beyond the ranges for n and dn by calling this 
                    function recursively. If set to None (default), the search 
                    will not be extended. Otherwise, search will extend along n
                    and/or dn if in-range frequencies are found at the 
                    specified boundaries of n and dn. This parameter must be
                    specified as a dictionary with the following keys and
                    values:
                    'n'     [None or list] If set to None, do not extend search
                            for more values of n. Otherwise it must be a list
                            containing one or both of the strings 'up' and
                            'down'. If 'up' is present, extend search for 
                            higher values of n from the previous iteration. If
                            'down' is present in the list, extend search for
                            values of n lower than specified in the range in 
                            previous iteration.
                    'dn'    [None or list] If set to None, do not extend search
                            for more values of dn. Otherwise it must be a list
                            containing one or both of the strings 'up' and
                            'down'. If 'up' is present, extend search for 
                            higher values of dn from the previous iteration. If
                            'down' is present in the list, extend search for
                            values of dn lower than specified in the range in 
                            previous iteration.

    Output:

    Tuple of (n, dn, freq) where each of the elements in the tuple is an array
    such that the transitions of combinations of n and dn produces 
    recombination lines for a given redshift in the specified frequency range.
    freq will be returned as an instance of class astropy.units.Quantity
    ---------------------------------------------------------------------------
    """

    try:
        freq_min, freq_max, atomic_number, atomic_mass
    except NameError:
        raise NameError('Inputs freq_min, freq_max, atomic_number, atomic_mass must be specified')

    if not isinstance(n_min, int):
        raise TypeError('Input n_min must be an integer')
    if n_min < 1:
        raise ValueError('Input n_min must be greater than 1')

    if not isinstance(n_max, int):
        raise TypeError('Input n_max must be an integer')
    if n_max < n_min:
        raise ValueError('Input n_max must be greater than n_min')

    if not isinstance(dn_min, int):
        raise TypeError('Input dn_min must be an integer')
    if dn_min < 1:
        raise ValueError('Input dn_min must be greater than 1')

    if not isinstance(dn_max, int):
        raise TypeError('Input dn_max must be an integer')
    if dn_max < dn_min:
        raise ValueError('Input dn_max must be greater than dn_min')

    if not isinstance(z, (int,float)):
        if isinstance(z, NP.ndarray):
            if z.size != 1:
                raise TypeError('Input z must be a scalar')
        else:
            raise TypeError('Input z must be a scalar')
            
    if not isinstance(freq_min, (int,float,units.Quantity)):
        raise TypeError('Input freq_min must be a scalar')
    if not isinstance(freq_min, units.Quantity):
        freq_min = freq_min * units.Hertz
    if freq_min <= 0.0 * units.Hertz:
        raise ValueError('Input freq_min must be positive')

    if not isinstance(freq_max, (int,float,units.Quantity)):
        raise TypeError('Input freq_max must be a scalar')
    if not isinstance(freq_max, units.Quantity):
        freq_max = freq_max * units.Hertz
    if freq_max <= freq_min:
        raise ValueError('Input freq_max must be greater than freq_min')

    if extendsearch is not None:
        if not isinstance(extendsearch, dict):
            raise TypeError('Input extendsearch must be a dictionary')
        for key in extendsearch:
            if extendsearch[key] is not None:
                if not isinstance(extendsearch[key], list):
                    raise TypeError('Value under key {0} of input dictionary extendsearch must be a list'.format(key))

    nvect = NP.arange(n_min, n_max+1)
    dnvect = NP.arange(dn_min, dn_max+1)
    ngrid, dngrid = NP.meshgrid(nvect, dnvect, indexing='ij')
    nu = redshifted_freq_recomb(atomic_number, atomic_mass, ngrid.reshape(-1), dngrid.reshape(-1), z=z, screening=screening)
    nu = nu.reshape(nvect.size, dnvect.size, -1)
    ind_select = NP.where(NP.logical_and(nu >= freq_min, nu <= freq_max))
    nu_select = nu[ind_select]
    n_select = ngrid[:,:,NP.newaxis][ind_select]
    dn_select = dngrid[:,:,NP.newaxis][ind_select]
    nu_in_range = None
    n_in_range = None
    dn_in_range = None
    if nu_select.size > 0:
        if nu_in_range is not None:
            nu_in_range = units.Quantity(NP.concatenate((nu_in_range.value, nu_select.value)), nu_select.unit)
            n_in_range = NP.concatenate((n_in_range, n_select))
            dn_in_range = NP.concatenate((dn_in_range, dn_select))
        else:
            nu_in_range = nu_select.copy()
            n_in_range = NP.copy(n_select)
            dn_in_range = NP.copy(dn_select)

        if extendsearch is not None:
            new_extendsearch = None
            for key in extendsearch:
                if extendsearch[key] is not None:
                    if key == 'n':
                        if n_select.max() == n_max:
                            if 'up' in extendsearch[key]:
                                new_n_min = n_max + 1
                                new_n_max = 2 * n_max + 1 - n_min
                                if new_extendsearch is None:
                                    new_extendsearch = {key: ['up']}
                                elif key not in new_extendsearch:
                                    new_extendsearch[key] = ['up']
                                else:
                                    new_extendsearch[key] += ['up']
                                new_n_select, new_dn_select, new_nu_select = search_transitions_in_freq_range(freq_min, freq_max, atomic_number, atomic_mass, n_min=new_n_min, n_max=new_n_max, dn_min=dn_min, dn_max=dn_max, z=z, screening=screening, extendsearch=new_extendsearch)
                                if new_nu_select.size > 0:
                                    if nu_in_range is not None:
                                        nu_in_range = units.Quantity(NP.concatenate((nu_in_range.value, new_nu_select.value)), new_nu_select.unit)
                                        n_in_range = NP.concatenate((n_in_range, new_n_select))
                                        dn_in_range = NP.concatenate((dn_in_range, new_dn_select))
                                    else:
                                        nu_in_range = new_nu_select.copy()
                                        n_in_range = NP.copy(new_n_select)
                                        dn_in_range = NP.copy(new_dn_select)
                        if n_select.min() == n_min:
                            if 'down' in extendsearch[key]:
                                if n_min > 1:
                                    new_n_min = max([1, 2*n_min - n_max - 1])
                                    new_n_max = n_max - 1
                                    if new_extendsearch is None:
                                        new_extendsearch = {key: ['down']}
                                    elif key not in new_extendsearch:
                                        new_extendsearch[key] = ['down']
                                    else:
                                        new_extendsearch[key] += ['down']
                                    new_n_select, new_dn_select, new_nu_select = search_transitions_in_freq_range(freq_min, freq_max, atomic_number, atomic_mass, n_min=new_n_min, n_max=new_n_max, dn_min=dn_min, dn_max=dn_max, z=z, screening=screening, extendsearch=new_extendsearch)
                                    if new_nu_select.size > 0:
                                        if nu_in_range is not None:
                                            nu_in_range = units.Quantity(NP.concatenate((new_nu_select.value, nu_in_range.value)), new_nu_select.unit)
                                            n_in_range = NP.concatenate((new_n_select, n_in_range))
                                            dn_in_range = NP.concatenate((new_dn_select, dn_in_range))
                                        else:
                                            nu_in_range = new_nu_select.copy()
                                            n_in_range = NP.copy(new_n_select)
                                            dn_in_range = NP.copy(new_dn_select)
                    if key == 'dn':
                        if dn_select.max() == dn_max:
                            if 'up' in extendsearch[key]:
                                new_dn_min = dn_max + 1
                                new_dn_max = 2 * dn_max + 1 - dn_min
                                if new_extendsearch is None:
                                    new_extendsearch = {key: ['up']}
                                elif key not in new_extendsearch:
                                    new_extendsearch[key] = ['up']
                                else:
                                    new_extendsearch[key] += ['up']
                                new_n_select, new_dn_select, new_nu_select = search_transitions_in_freq_range(freq_min, freq_max, atomic_number, atomic_mass, n_min=n_min, n_max=n_max, dn_min=new_dn_min, dn_max=new_dn_max, z=z, screening=screening, extendsearch=new_extendsearch)
                                if new_nu_select.size > 0:
                                    if nu_in_range is not None:
                                        nu_in_range = units.Quantity(NP.concatenate((nu_in_range.value, new_nu_select.value)), new_nu_select.unit)
                                        n_in_range = NP.concatenate((n_in_range, new_n_select))
                                        dn_in_range = NP.concatenate((dn_in_range, new_dn_select))
                                    else:
                                        nu_in_range = new_nu_select.copy()
                                        n_in_range = NP.copy(new_n_select)
                                        dn_in_range = NP.copy(new_dn_select)
                        if dn_select.min() == dn_min:
                            if 'down' in extendsearch[key]:
                                if dn_min > 1:
                                    new_dn_min = max([1, 2*dn_min - dn_max - 1])
                                    new_dn_max = dn_max - 1
                                    if new_extendsearch is None:
                                        new_extendsearch = {key: ['down']}
                                    elif key not in new_extendsearch:
                                        new_extendsearch[key] = ['down']
                                    else:
                                        new_extendsearch[key] += ['down']
                                    new_n_select, new_dn_select, new_nu_select = search_transitions_in_freq_range(freq_min, freq_max, atomic_number, atomic_mass, n_min=n_min, n_max=n_max, dn_min=new_dn_min, dn_max=new_dn_max, z=z, screening=screening, extendsearch=new_extendsearch)
                                    if new_nu_select.size > 0:
                                        if nu_in_range is not None:
                                            nu_in_range = units.Quantity(NP.concatenate((new_nu_select.value, nu_in_range.value)), new_nu_select.unit)
                                            n_in_range = NP.concatenate((new_n_select, n_in_range))
                                            dn_in_range = NP.concatenate((new_dn_select, dn_in_range))
                                        else:
                                            nu_in_range = new_nu_select.copy()
                                            n_in_range = NP.copy(new_n_select)
                                            dn_in_range = NP.copy(new_dn_select)
                            
    return (n_in_range, dn_in_range, nu_in_range)
    
###############################################################################
    
