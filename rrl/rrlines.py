import numpy as NP
from astropy import constants as FCNST

def restframe_freq_recomb(atomic_number, atomic_mass, n, dn=1, screening=False):

    try:
        atomic_number, atomic_mass, n
    except NameError:
        raise NameError('Inputs atmoic number, atomic mass and lower electron level must be specified')

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
        raise TypeError('Input dn must be an integer or numpy array')
    dn = NP.asarray(dn).reshape(-1)
    if NP.any(dn < 1):
        raise ValueError('Lower electron level must be greater than 1')
    if dn.size != n.size:
        raise ValueError('Sizes of inputs n and dn must be same')

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

def redshifted_freq_recomb(atomic_number, atomic_mass, n, dn=1, z=0.0, screening=False):

    nu = restframe_freq_recomb(atomic_number, atomic_mass, n, dn=dn, screening=screening)
    if not isinstance(z, (int, float, NP.ndarray)):
        raise TypeError('Input must be a scalar or a numpy array')
    else:
        z = NP.asarray(z).reshape(1,-1)
        nu = nu.reshape(-1,1) / (1.0 + z)
        return nu

def check_recomb_transitions_in_freq_range(freq_min, freq_max, atomic_number, atomic_mass, n, dn, z=None, screening=False):

    try:
        freq_min, freq_max, atomic_number, atomic_mass, n, dn
    except NameError:
        raise NameError('Inputs freq_min, freq_max, atomic_number, atomic_mass, n, dn must be specified')

    if not isinstance(freq_min, (int,float)):
        raise TypeError('Input freq_min must be a scalar')
    if freq_min <= 0.0:
        raise ValueError('Input freq_min must be positive')

    if not isinstance(freq_max, (int,float)):
        raise TypeError('Input freq_max must be a scalar')
    if freq_max <= 0.0:
        raise ValueError('Input freq_max must be positive')

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
    
def search_transitions_in_freq_range(freq_min, freq_max, atomic_number, atomic_mass, n_min=1, n_max=1000, dn_min=1, dn_max=10, z=0.0, screening=False, extendsearch=None):

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
    ind_select = NP.where(NP.logical_and(nu.value >= freq_min, nu.value <= freq_max))
    nu_select = nu[ind_select]
    n_select = ngrid[:,:,NP.newaxis][ind_select]
    dn_select = dngrid[:,:,NP.newaxis][ind_select]
    nu_in_range = None
    n_in_range = None
    dn_in_range = None
    if nu_select.size > 0:
        if nu_in_range is not None:
            nu_in_range = NP.concatenate((nu_in_range, nu_select))
            n_in_range = NP.concatenate((n_in_range, n_select))
            dn_in_range = NP.concatenate((dn_in_range, dn_select))
        else:
            nu_in_range = NP.copy(nu_select)
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
                                        nu_in_range = NP.concatenate((nu_in_range, new_nu_select))
                                        n_in_range = NP.concatenate((n_in_range, new_n_select))
                                        dn_in_range = NP.concatenate((dn_in_range, new_dn_select))
                                    else:
                                        nu_in_range = NP.copy(new_nu_select)
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
                                            nu_in_range = NP.concatenate((new_nu_select, nu_in_range))
                                            n_in_range = NP.concatenate((new_n_select, n_in_range))
                                            dn_in_range = NP.concatenate((new_dn_select, dn_in_range))
                                        else:
                                            nu_in_range = NP.copy(new_nu_select)
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
                                        nu_in_range = NP.concatenate((nu_in_range, new_nu_select))
                                        n_in_range = NP.concatenate((n_in_range, new_n_select))
                                        dn_in_range = NP.concatenate((dn_in_range, new_dn_select))
                                    else:
                                        nu_in_range = NP.copy(new_nu_select)
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
                                            nu_in_range = NP.concatenate((new_nu_select, nu_in_range))
                                            n_in_range = NP.concatenate((new_n_select, n_in_range))
                                            dn_in_range = NP.concatenate((new_dn_select, dn_in_range))
                                        else:
                                            nu_in_range = NP.copy(new_nu_select)
                                            n_in_range = NP.copy(new_n_select)
                                            dn_in_range = NP.copy(new_dn_select)
                            
    return (n_in_range, dn_in_range, nu_in_range)
    
    
