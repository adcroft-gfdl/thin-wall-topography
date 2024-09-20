from GMesh import GMesh
import numpy

class StatsBase(object):
    """A base class for Stats
    Provides numpy-like itemized access with "view" (instead of copy) when possible.
    low, ave and hgh properties are not allocated during instantiation.
    """
    __slots__ = ('shape', '_low', '_ave', '_hgh')
    def __init__(self, shape=None, low=None, ave=None, hgh=None):
        self.shape = shape
        self._ave = None
        self._low = None
        self._hgh = None
        if shape is None:
            if (ave is not None and low is not None and ave.shape != low.shape) or \
               (ave is not None and hgh is not None and ave.shape != hgh.shape) or \
               (low is not None and hgh is not None and low.shape != hgh.shape):
                raise ValueError("Shapes of ave, low, and hgh must be consistent.")
            if low is not None:
                self.shape = low.shape
            elif ave is not None:
                self.shape = ave.shape
            elif hgh is not None:
                self.shape = hgh.shape
        if low is not None: self.low = low
        if ave is not None: self.ave = ave
        if hgh is not None: self.hgh = hgh
    @property
    def low(self):
        if self._low is None:
            self._low = numpy.zeros(self.shape)
        return self._low
    @low.setter
    def low(self, value):
        assert value.shape==self.shape, "Wrong shape for 'low'!"
        self._low = value
    @property
    def ave(self):
        if self._ave is None:
            self._ave = numpy.zeros(self.shape)
        return self._ave
    @ave.setter
    def ave(self, value):
        assert value.shape==self.shape, "Wrong shape for 'ave'!"
        self._ave = value
    @property
    def hgh(self):
        if self._hgh is None:
            self._hgh = numpy.zeros(self.shape)
        return self._hgh
    @hgh.setter
    def hgh(self, value):
        assert value.shape==self.shape, "Wrong shape for 'hgh'!"
        self._hgh = value
    def __getitem__(self, key):
        return StatsBase(low=self.low[key], ave=self.ave[key], hgh=self.hgh[key])
    def __setitem__(self, key, value):
        self.low[key] = value.low
        self.hgh[key] = value.hgh
        self.ave[key] = value.ave

class Stats(StatsBase):
    """Container for statistics fields

    shape - shape of these arrays
    low   - minimum value
    hgh   - maximum value
    ave   - mean value
    """
    def __init__(self, shape, low=None, ave=None, hgh=None):
        assert len(shape)==2, "Shape size error. Stats object needs to be strictly two-dimensional."
        StatsBase.__init__(self, shape=shape, low=low, ave=ave, hgh=hgh)
        if (low is not None) and (ave is not None) and (hgh is not None):
            self.set(low, hgh, ave)
        else:
            if ave is not None: self.set_equal(ave)
            if low is not None: self.set_equal(low)
            if hgh is not None: self.set_equal(hgh)
    def __repr__(self):
        return '<Stats shape:(%i,%i)>'%(self.shape[0], self.shape[1])
    def __copy__(self):
        return Stats(self.shape, ave=self.ave, low=self.low, hgh=self.hgh)
    def copy(self):
        """Returns new instance with copied values"""
        return self.__copy__()
    def dump(self):
        print('min:')
        print(self.low)
        print('mean:')
        print(self.ave)
        print('max:')
        print(self.hgh)
    def set_equal(self, values):
        assert values.shape == self.shape, 'Data has the wrong shape!'
        self.ave = values.copy()
        self.low = values.copy()
        self.hgh = values.copy()
    def set(self, min, max, mean):
        assert min.shape == self.shape, 'Min data has the wrong shape!'
        assert max.shape == self.shape, 'Max data has the wrong shape!'
        assert mean.shape == self.shape, 'Mean data has the wrong shape!'
        self.ave = mean.copy()
        self.low = min.copy()
        self.hgh = max.copy()
    def mean4(self):
        """Return 2d/4-point mean"""
        return 0.25*( (self.ave[::2,::2]+self.ave[1::2,1::2]) + (self.ave[::2,1::2]+self.ave[1::2,::2]) )
    def min4(self):
        """Return 2d/4-point minimum"""
        return numpy.minimum( numpy.minimum( self.low[::2,::2], self.low[1::2,1::2]),
                              numpy.minimum( self.low[::2,1::2], self.low[1::2,::2]) )
    def max4(self):
        """Return 2d/4-point maximum"""
        return numpy.maximum( numpy.maximum( self.hgh[::2,::2], self.hgh[1::2,1::2]),
                              numpy.maximum( self.hgh[::2,1::2], self.hgh[1::2,::2]) )
    def mean2u(self):
        """Return 2d/2-point mean on u-edges"""
        return 0.5*( self.ave[::2,::2] + self.ave[1::2,::2] )
    def min2u(self):
        """Return 2d/2-point minimum on u-edges"""
        return numpy.minimum( self.low[::2,::2], self.low[1::2,::2] )
    def max2u(self):
        """Return 2d/2-point maximum on u-edges"""
        return numpy.maximum( self.hgh[::2,::2], self.hgh[1::2,::2] )
    def mean2v(self):
        """Return 2d/2-point mean on v-edges"""
        return 0.5*( self.ave[::2,::2] + self.ave[::2,1::2] )
    def min2v(self):
        """Return 2d/2-point minimum on v-edges"""
        return numpy.minimum( self.low[::2,::2], self.low[::2,1::2] )
    def max2v(self):
        """Return 2d/2-point maximum on v-edges"""
        return numpy.maximum( self.hgh[::2,::2], self.hgh[::2,1::2] )
    def flip(self, axis):
        """Flip the data along the given axis"""
        self.low = numpy.flip(self.low, axis=axis)
        self.ave = numpy.flip(self.ave, axis=axis)
        self.hgh = numpy.flip(self.hgh, axis=axis)
    def transpose(self):
        """Transpose data swapping i-j indexes"""
        self.shape = (self.shape[1], self.shape[0])
        self.low = self.low.T
        self.ave = self.ave.T
        self.hgh = self.hgh.T

def od(dir):
    """Returns the name of the opposite direction"""
    op_map = {'S': 'N', 'N': 'S', 'W': 'E', 'E': 'W'}
    return op_map.get(dir, None)
def nd(dir):
    """Returns the name of the neighboring direction"""
    nb_map = {'S': ('W','E'), 'N': ('E','W'), 'W': ('N','S'), 'E': ('S','N')}
    return nb_map.get(dir, None)
def intercard(dir):
    """Returns the correct name of the intercardinal directions"""
    ordinals = ['SW', 'SE', 'NW', 'NE']
    return dir*(dir in ordinals) + dir[::-1]*(dir[::-1] in ordinals)
def dmajor(edge_name):
    """Returns the name of "major" direction of an exterior edge section.
    E.g. dmajor('NNW') = 'N', dmajor('WNW') = 'W'
    """
    assert len(edge_name)==3
    assert len(edge_name)==3
    first = (edge_name[0] in edge_name[1:])
    last = (edge_name[-1] in edge_name[:-1])
    return edge_name[0]*(first) + edge_name[-1]*(not first and last)
def dminor(edge_name):
    """Returns the name of "minor" direction of an exterior edge section.
    E.g. dmajor('NNW') = 'W', dmajor('WNW') = 'N'
    """
    assert len(edge_name)==3
    return edge_name.replace(dmajor(edge_name),'')
def secintercard(dir):
    """Returns the correct name of the secondary intercardinal directions"""
    return dmajor(dir) + intercard(dmajor(dir)+dminor(dir))
def exterior_edges(dir):
    """Returns the name of two exterior edge sections at the given direction.
    """
    nds = nd(dir)
    return (secintercard(nds[0]+dir*2), secintercard(nds[1]+dir*2))
def max_Stats(e1, e2):
    """Returns a StatsBase object that contains higher stats of the two"""
    low = numpy.maximum(e1.low, e2.low)
    ave = numpy.maximum(e1.ave, e2.ave)
    hgh = numpy.maximum(e1.hgh, e2.hgh)
    return StatsBase(ave=ave, low=low, hgh=hgh)

class ThinWalls(GMesh):
    """Container for thin wall topographic data and mesh.

    Additional members:
    c_simple - elevation statistics of cell, shape (nj,ni)
    u_simple - elevation statistics of western edge of cell, shape (nj,ni+1)
    v_simple - elevation statistics of southern edge of cell, shape (nj+1,nj)
    shapeu  - shape of zu_simple_mean, ie. =(nj,ni+1)
    shapev  - shape of zv_simple_mean, ie. =(nj+1,ni)

    Extends the GMesh class.
    """

    def __init__(self, *args, **kwargs):
        """Constructor for ThinWalls."""
        GMesh.__init__(self, *args, **kwargs)
        self.shapeu = (self.nj, self.ni+1)
        self.shapev = (self.nj+1, self.ni)
        self.c_simple = Stats(self.shape)
        self.u_simple = Stats(self.shapeu)
        self.v_simple = Stats(self.shapev)
        self.c_effective = Stats(self.shape)
        self.u_effective = Stats(self.shapeu)
        self.v_effective = Stats(self.shapev)
    def __copy__(self):
        copy = ThinWalls(shape=self.shape, lon=self.lon, lat=self.lat)
        copy.c_simple = self.c_simple.copy()
        copy.u_simple = self.u_simple.copy()
        copy.v_simple = self.v_simple.copy()
        copy.c_effective = self.c_effective.copy()
        copy.u_effective = self.u_effective.copy()
        copy.v_effective = self.v_effective.copy()
        return copy
    def copy(self):
        """Returns new instance with copied values"""
        return self.__copy__()
    def transpose(self):
        """Transpose data swapping i-j indexes"""
        super().transpose()
        self.c_simple.transpose()
        self.u_simple.transpose()
        self.v_simple.transpose()
        self.u_simple, self.v_simple = self.v_simple, self.u_simple
        self.c_effective.transpose()
        self.u_effective.transpose()
        self.v_effective.transpose()
        self.u_effective, self.v_effective = self.v_effective, self.u_effective
        self.shape = self.c_effective.shape
        self.shapeu = self.u_effective.shape
        self.shapev = self.v_effective.shape
    def refine(self):
        """Returns new ThinWalls instance with twice the resolution."""
        M = super().refineby2()
        return ThinWalls(lon=M.lon, lat=M.lat)
    def dump(self):
        """Dump Mesh to tty."""
        super().dump()
        self.c_simple.dump()
        self.u_simple.dump()
        self.v_simple.dump()
        self.c_effective.dump()
        self.u_effective.dump()
        self.v_effective.dump()
    def set_cell_mean(self, data):
        """Set elevation of cell center."""
        assert data.shape==self.shape, 'data argument has wrong shape'
        self.c_simple.set_equal(data)
    def set_edge_mean(self, datau, datav):
        """Set elevation of cell edges u,v."""
        assert datau.shape==self.shapeu, 'datau argument has wrong shape'
        assert datav.shape==self.shapev, 'datav argument has wrong shape'
        self.u_simple.set_equal(datau)
        self.v_simple.set_equal(datav)
    def init_effective_values(self):
        """Initialize effective values by setting equal to simple values."""
        self.c_effective = self.c_simple.copy()
        self.u_effective = self.u_simple.copy()
        self.v_effective = self.v_simple.copy()
    def set_edge_to_step(self, method='max'):
        """Set elevation of cell edges to step topography."""
        tmp = numpy.zeros(self.shapeu)
        if method=='max':
            tmp[:,1:-1] = numpy.maximum( self.c_simple.ave[:,:-1], self.c_simple.ave[:,1:] )
        elif method=='min':
            tmp[:,1:-1] = numpy.maximum( self.c_simple.ave[:,:-1], self.c_simple.ave[:,1:] )
        elif method=='ave':
            tmp[:,1:-1] = 0.5 * ( self.c_simple.ave[:,:-1], self.c_simple.ave[:,1:] )
        tmp[:,0] = self.c_simple.ave[:,0]
        tmp[:,-1] = self.c_simple.ave[:,-1]
        self.u_simple.set_equal( tmp )
        tmp = numpy.zeros(self.shapev)
        tmp[1:-1,:] = numpy.maximum( self.c_simple.ave[:-1,:], self.c_simple.ave[1:,:] )
        tmp[0,:] = self.c_simple.ave[0,:]
        tmp[-1,:] = self.c_simple.ave[-1,:]
        self.v_simple.set_equal( tmp )
    def set_center_from_corner(self, data):
        """Set elevation of cell centers from corners."""
        self.c_simple.ave = 0.25 * ( (data[:-1,:-1] + data[1:,1:])
                                    +(data[1:,:-1] + data[:-1,1:]) )
        self.c_simple.hgh = numpy.maximum( numpy.maximum( data[:-1,:-1], data[1:,1:]),
                                           numpy.maximum( data[1:,:-1], data[:-1,1:]) )
        self.c_simple.low = numpy.minimum( numpy.minimum( data[:-1,:-1], data[1:,1:]),
                                           numpy.minimum( data[1:,:-1], data[:-1,1:]) )
    def set_edge_from_corner(self, data):
        """Set elevation of cell edges from corners."""
        self.u_simple.ave = 0.5 * ( data[:-1,:] + data[1:,:] )
        self.u_simple.hgh = numpy.maximum( data[:-1,:], data[1:,:] )
        self.u_simple.low = numpy.minimum( data[:-1,:], data[1:,:] )
        self.v_simple.ave = 0.5 * ( data[:,:-1] + data[:,1:] )
        self.v_simple.hgh = numpy.maximum( data[:,:-1], data[:,1:] )
        self.v_simple.low = numpy.minimum( data[:,:-1], data[:,1:] )
    def sec(self, direction, measure='effective'):
        """
        Returns a StatsBase object that is a view of the heights at various locations.

        Key map:

         ----NNW-----NNE----
         |        |        |
        WNW  NW   N   NE  ENE
         |        |        |
         -----W-------E-----
         |        |        |
        WSW  SW   S   SE  ESE
         |        |        |
         ----SSW-----SSE----
        """
        if measure == 'effective':
            C, U, V = self.c_effective, self.u_effective, self.v_effective
        elif measure == 'simple':
            C, U, V = self.c_simple, self.u_simple, self.v_simple
        else:
            raise Exception('Measure error')

        seclist = {'S': U[0::2, 1::2], 'N': U[1::2, 1::2], 'W': V[1::2, 0::2], 'E': V[1::2, 1::2],
                   'SW': C[0::2, 0::2], 'SE': C[0::2, 1::2], 'NW': C[1::2, 0::2], 'NE': C[1::2, 1::2],
                   'SSW': V[0:-1:2, 0::2], 'SSE': V[0:-1:2, 1::2], 'NNW': V[2::2, 0::2], 'NNE': V[2::2, 1::2],
                   'WSW': U[0::2, 0:-1:2], 'WNW': U[1::2, 0:-1:2], 'ESE': U[0::2, 2::2], 'ENE': U[1::2, 2::2]}
        assert direction in seclist.keys(), '{:} not in section list'.format(direction)

        return seclist[direction]
    def diagnose_pathways_straight(self):
        ssw_to_nnw, sse_to_nne, ssw_to_nne, sse_to_nnw = self.diagnose_pathways('SN')
        sn = numpy.minimum( numpy.minimum( ssw_to_nnw, ssw_to_nne), numpy.minimum( sse_to_nnw, sse_to_nne))

        wnw_to_ene, wsw_to_ese, wnw_to_ese, wsw_to_ene = self.diagnose_pathways('WE')
        we = numpy.minimum( numpy.minimum( wnw_to_ene, wnw_to_ese), numpy.minimum( wsw_to_ene, wsw_to_ese))
        return {'SN':sn, 'WE':we}
    def diagnose_pathways_corner(self):
        ssw_to_ene, sse_to_ese, ssw_to_ese, sse_to_ene = self.diagnose_pathways('SE')
        se = numpy.minimum( numpy.minimum( ssw_to_ene, sse_to_ese), numpy.minimum( ssw_to_ese, sse_to_ene))

        ese_to_nnw, ene_to_nne, ese_to_nne, ene_to_nnw = self.diagnose_pathways('EN')
        en = numpy.minimum( numpy.minimum( ese_to_nnw, ene_to_nne), numpy.minimum( ese_to_nne, ene_to_nnw))

        nne_to_wsw, nnw_to_wnw, nne_to_wnw, nnw_to_wsw = self.diagnose_pathways('NW')
        nw = numpy.minimum( numpy.minimum( nne_to_wsw, nnw_to_wnw), numpy.minimum( nne_to_wnw, nnw_to_wsw))

        wnw_to_sse, wsw_to_ssw, wnw_to_ssw, wsw_to_sse = self.diagnose_pathways('WS')
        ws = numpy.minimum( numpy.minimum( wnw_to_sse, wsw_to_ssw), numpy.minimum( wnw_to_ssw, wsw_to_sse))
        return {'SE':se, 'EN':en, 'NW':nw, 'WS':ws}
    def diagnose_pathways(self, path):
        """Returns the four connectivities along a given path
        Parameters
        ----------
        path : str
            Straight pathways (NS and EW) and corner pathways (SE, EN, NW, WS). Note that for corner
            pathways, `path` argument needs to be the exact form from the listed four, as it is constrained
            by the order of the two section names from module method `nd`.
        Output
        ----------
        d1a_to_d2b, d1b_to_d2a, d1a_to_d2a, d1b_to_d2b : array of float
            Minimun connectivity (maximum of low along a certain path) of all possible pathways
        """
        def straight_side(sec1, sec2):
            assert dminor(sec1)==dminor(sec2)
            route1 = self.sec(dminor(sec1)).low
            route2 = numpy.maximum(numpy.maximum(self.sec(dmajor(sec1)).low, self.sec(dmajor(sec2)).low),
                                   self.sec(od(dminor(sec1))).low)
            inner = numpy.minimum(route1, route2)
            return numpy.maximum(numpy.maximum(self.sec(sec1).low, self.sec(sec2).low), inner)

        def straight_diag(sec1, sec2):
            assert dminor(sec1)==od(dminor(sec2))
            route1 = numpy.maximum(self.sec(dmajor(sec1)).low, self.sec(dminor(sec2)).low)
            route2 = numpy.maximum(self.sec(dmajor(sec2)).low, self.sec(dminor(sec1)).low)
            inner = numpy.minimum(route1, route2)
            return numpy.maximum(numpy.maximum(self.sec(sec1).low, self.sec(sec2).low), inner)

        def corner_nrw(sec1, sec2):
            assert dmajor(sec1)==dminor(sec2) and dmajor(sec2)==dminor(sec1)
            return numpy.maximum(self.sec(sec1).low, self.sec(sec2).low)

        def corner_mid(sec1, sec2):
            assert dmajor(sec1)==dminor(sec2)
            route1 = self.sec(dmajor(sec1)).low
            route2 = numpy.maximum(numpy.maximum(self.sec(dmajor(sec2)).low, self.sec(od(dmajor(sec2))).low),
                                   self.sec(od(dmajor(sec1))).low)
            inner = numpy.minimum(route1, route2)
            return numpy.maximum(numpy.maximum(self.sec(sec1).low, self.sec(sec2).low), inner)

        def corner_wid(sec1, sec2):
            assert dmajor(sec1)==od(dminor(sec2)) and dmajor(sec2)==od(dminor(sec1))
            route1 = numpy.maximum(self.sec(dmajor(sec1)).low, self.sec(dmajor(sec2)).low)
            route2 = numpy.maximum(self.sec(dminor(sec1)).low, self.sec(dminor(sec2)).low)
            inner = numpy.minimum(route1, route2)
            return numpy.maximum(numpy.maximum(self.sec(sec1).low, self.sec(sec2).low), inner)

        d1, d2 = path
        d1a, d1b = exterior_edges(d1)
        d2a, d2b = exterior_edges(d2)

        if d1==od(d2):
            d1a_to_d2b = straight_side(d1a, d2b)
            d1b_to_d2a = straight_side(d1b, d2a)
            d1a_to_d2a = straight_diag(d1a, d2a)
            d1b_to_d2b = straight_diag(d1b, d2b)
        else:
            d1b_to_d2a = corner_nrw(d1b, d2a)
            d1a_to_d2a = corner_mid(d1a, d2a)
            d1b_to_d2b = corner_mid(d2b, d1b)
            d1a_to_d2b = corner_wid(d1a, d2b)

        return d1a_to_d2b, d1b_to_d2a, d1a_to_d2a, d1b_to_d2b

    def push_interior_corners(self, adjust_centers=True, matlab=False, verbose=False):
        """"A wrapper for push out high corners"""
        idx_sw, corner_sw = self.find_interior_corner('SW', adjust_centers=adjust_centers, matlab=matlab)
        idx_se, corner_se = self.find_interior_corner('SE', adjust_centers=adjust_centers, matlab=matlab)
        idx_nw, corner_nw = self.find_interior_corner('NW', adjust_centers=adjust_centers, matlab=matlab)
        idx_ne, corner_ne = self.find_interior_corner('NE', adjust_centers=adjust_centers, matlab=matlab)

        if verbose:
            print('push_interior_corners')
            print("  SW corner pushed: {}".format(idx_sw[0].size))
            print("  NW corner pushed: {}".format(idx_nw[0].size))
            print("  NE corner pushed: {}".format(idx_ne[0].size))
            print("  SE corner pushed: {}".format(idx_se[0].size))

        self.push_interior_corner('SW', idx_sw, corner_sw)
        self.push_interior_corner('SE', idx_se, corner_se)
        self.push_interior_corner('NW', idx_nw, corner_nw)
        self.push_interior_corner('NE', idx_ne, corner_ne)

    def push_interior_corner(self, dir, idx, corner):
        assert dir[0]=='S' or dir[0]=='N'
        assert dir[1]=='W' or dir[1]=='E'

        E1, E2 = self.sec(dir[0]+dir), self.sec(dir[1]+dir)

        E1[idx] = max_Stats(E1[idx], corner)
        E2[idx] = max_Stats(E2[idx], corner)

    def find_interior_corner(self, dir, adjust_centers=True, matlab=False):
        """Finds out if corner is the highest ridge."""
        assert dir[0]=='S' or dir[0]=='N'
        assert dir[1]=='W' or dir[1]=='E'

        R1, R2 = self.sec(dir[0]), self.sec(dir[1]) # Target corner sections
        B1, B2 = self.sec(od(dir[0])), self.sec(od(dir[1])) # Opposing corner sections
        C = self.sec(dir) # Targer corner cell centers

        inner = StatsBase(low=numpy.minimum(R1.low, R2.low),
                          ave=0.5*(R1.ave+R2.ave),
                          hgh=numpy.maximum(R1.hgh, R2.hgh))
        opp_ridge = numpy.maximum(B1.low, B2.low)
        idx = numpy.nonzero( inner.low>opp_ridge )

        # Adjust inner edges and cell centers
        R1.low[idx], R2.low[idx] = opp_ridge[idx], opp_ridge[idx]
        if adjust_centers:
            opp_mean = (  self.sec(od(dir[0])+dir[1]).ave
                        + self.sec(dir[0]+od(dir[1])).ave
                        + self.sec(od(dir[0])+od(dir[1])).ave )/3.0
            C.low[idx] = opp_ridge[idx]
            if matlab:
                C.ave[idx] = opp_mean[idx]
                C.hgh[idx] = opp_ridge[idx]
            else:
                C.ave[idx] = numpy.maximum(C.ave[idx], opp_mean[idx])
                C.hgh[idx] = numpy.maximum(C.hgh[idx], opp_ridge[idx])
        if matlab:
            update_interior_mean_max = False
        else:
            update_interior_mean_max = True
        if update_interior_mean_max:
            R1.ave[idx], R2.ave[idx] = opp_ridge[idx], opp_ridge[idx] # HW ???
            R1.hgh[idx], R2.hgh[idx] = opp_ridge[idx], opp_ridge[idx]

        return idx, inner[idx]

    def lower_interior_buttresses(self, do_ave=True, adjust_mean=False, verbose=False):
        """A wrapper to find and remove the tallest inner edge at all directions"""
        if verbose:
            print('lower_interior_buttresses')
        if do_ave:
            adjust_mean = False
        for dir in ['S', 'N', 'W', 'E']:
            if do_ave:
                idx, idx_ave = self.find_interior_buttress(dir, do_ave=do_ave, adjust_mean=adjust_mean)
            else:
                idx = self.find_interior_buttress(dir, adjust_mean=adjust_mean)
            if verbose:
                print("  {:} buttress removed (low): {:}".format(dir, idx[0].size))
                if do_ave:
                    print("  {:} buttress removed (ave): {:}".format(dir, idx_ave[0].size))
    def find_interior_buttress(self, dir, do_ave=True, adjust_mean=False):
        """Find and remove the tallest inner edge"""
        R = self.sec(dir)
        nds = nd(dir)
        B1, B2, B3 = self.sec(nds[0]), self.sec(nds[1]), self.sec(od(dir))

        # Low
        oppo3 = numpy.maximum(numpy.maximum(B1.low, B2.low), B3.low)
        idx = numpy.nonzero(R.low > oppo3)
        R.low[idx] = oppo3[idx]
        if adjust_mean:
            R.ave[idx] = numpy.maximum(numpy.maximum(B1.ave[idx], B2.ave[idx]), B3.ave[idx])

        # Ave
        if do_ave:
            oppo3 = numpy.maximum(numpy.maximum(B1.ave, B2.ave), B3.ave)
            idx_ave = numpy.nonzero(R.ave > oppo3)
            R.ave[idx_ave] = oppo3[idx_ave]
            return idx, idx_ave
        else:
            return idx

    def fold_interior_ridges(self, adjust_centers=False, adjust_low_only=False, verbose=False):
        """A wrapper to fold out ridges in all directions"""
        idx_s, ridge_s = self.find_interior_ridge('S', adjust_centers=adjust_centers, adjust_low_only=adjust_low_only)
        idx_n, ridge_n = self.find_interior_ridge('N', adjust_centers=adjust_centers, adjust_low_only=adjust_low_only)
        idx_w, ridge_w = self.find_interior_ridge('W', adjust_centers=adjust_centers, adjust_low_only=adjust_low_only)
        idx_e, ridge_e = self.find_interior_ridge('E', adjust_centers=adjust_centers, adjust_low_only=adjust_low_only)

        idx_ns, ridge_ns = self.find_interior_ridge('S', equal=True, adjust_centers=adjust_centers)
        idx_ew, ridge_ew = self.find_interior_ridge('W', equal=True, adjust_centers=adjust_centers)

        if verbose:
            print("  S: {}".format(idx_s[0].size))
            print("  N: {}".format(idx_n[0].size))
            print("  W: {}".format(idx_w[0].size))
            print("  E: {}".format(idx_e[0].size))
            print("  NS: {}".format(idx_ns[0].size))
            print("  EW: {}".format(idx_ew[0].size))

        self.fold_interior_ridge('S', idx_s, ridge_s, adjust_low_only=adjust_low_only)
        self.fold_interior_ridge('N', idx_n, ridge_n, adjust_low_only=adjust_low_only)
        self.fold_interior_ridge('W', idx_w, ridge_w, adjust_low_only=adjust_low_only)
        self.fold_interior_ridge('E', idx_e, ridge_e, adjust_low_only=adjust_low_only)
        self.fold_interior_ridge_equal('S', idx_ns, ridge_ns)
        self.fold_interior_ridge_equal('W', idx_ew, ridge_ew)

    def find_interior_ridge(self, dir, equal=False, adjust_centers=True, adjust_low_only=False):
        """Find high center ridges and adjust the inner edges (and cell centers)"""
        nd1, nd2 = nd(dir)
        # Ridge
        R1, R2 = self.sec(nd1), self.sec(nd2)
        # Buttresses at the targeting side (a) and opposing side (b) of the ridge
        Ba, Bb = self.sec(dir), self.sec(od(dir))
        # Cell centers at the two sides of the ridge
        Ca1, Ca2 = self.sec(intercard(dir+nd1)), self.sec(intercard(dir+nd2))
        Cb1, Cb2 = self.sec(intercard(od(dir)+nd2)), self.sec(intercard(od(dir)+nd1))
        # Outer edges parallel to the ridge
        Ea1, Ea2 = self.sec(secintercard(dir*2+nd1)), self.sec(secintercard(dir*2+nd2))
        Eb1, Eb2 = self.sec(secintercard(od(dir)*2+nd1)), self.sec(secintercard(od(dir)*2+nd2))

        central = StatsBase(low=numpy.minimum(R1.low, R2.low),
                            ave=0.5*(R1.ave+R2.ave),
                            hgh=numpy.maximum(R1.hgh, R2.hgh))
        oppos_low_min, oppos_low_max = numpy.minimum(Ba.low, Bb.low), numpy.maximum(Ba.low, Bb.low)

        ridges = ((central.low>oppos_low_min) & (central.low>=oppos_low_max))
        if equal:
            equal_sides = ((Ba.low == Bb.low) &
                           (Ca1.low+Ca2.low == Cb1.low+Cb2.low) & (Ea1.low+Ea2.low == Eb1.low+Eb2.low))
            idx = numpy.nonzero( ridges & equal_sides )
        else:
            # 1. Target side buttress is taller than its opposite
            high_buttress = (Ba.low > Bb.low)
            # 2. Equal buttresses. Target side cells are higher on average
            high_cell = ((Ba.low == Bb.low) & (Ca1.low+Ca2.low > Cb1.low+Cb2.low))
            # 3. Equal buttresses and cells. Target size outer edges are higher on average
            high_edge = ((Ba.low == Bb.low) &
                         (Ca1.low+Ca2.low == Cb1.low+Cb2.low) & (Ea1.low+Ea2.low > Eb1.low+Eb2.low))
            idx =  numpy.nonzero( ridges & (high_buttress | high_cell | high_edge) )

        # Adjust inner edges
        R1.low[idx], R2.low[idx] = oppos_low_min[idx], oppos_low_min[idx]
        Ba.low[idx] = oppos_low_min[idx]
        if equal:
            Bb.low[idx] = oppos_low_min[idx]

        # Adjust cell centers at the target side
        if adjust_centers:
            # This is the MatLab approach, seems wrong
            Ca1.low[idx], Ca2.low[idx] = oppos_low_min[idx], oppos_low_min[idx]
            if not adjust_low_only:
                Ca1.ave[idx], Ca2.ave[idx] = 0.5*(Cb1.ave[idx]+Cb2.ave[idx]), 0.5*(Cb1.ave[idx]+Cb2.ave[idx])
                Ca1.hgh[idx], Ca2.hgh[idx] = oppos_low_min[idx], oppos_low_min[idx]
            # The following is slightly different from the MatLab approach, which seems wrong.
            if equal:
              Cb1.low[idx], Cb2.low[idx] = oppos_low_min[idx], oppos_low_min[idx]
              if not adjust_low_only:
                  Cb1.ave[idx], Cb2.ave[idx] = 0.5*(Cb1.ave[idx]+Cb2.ave[idx]), 0.5*(Cb1.ave[idx]+Cb2.ave[idx])
                  Cb1.hgh[idx], Cb2.hgh[idx] = oppos_low_min[idx], oppos_low_min[idx]

        return idx, central[idx]

    def fold_interior_ridge(self, dir, idx, ridges, adjust_low_only=False):
        """Fold out ridge at the given direction"""
        nd1, nd2 = nd(dir)
        E1, E2 = self.sec(secintercard(dir*2+nd1)), self.sec(secintercard(dir*2+nd2))
        E3, E4 = self.sec(secintercard(dir+nd1*2)), self.sec(secintercard(dir+nd2*2))

        if adjust_low_only:
            E1.low[idx] = numpy.maximum(E1.low[idx], ridges.low)
            E2.low[idx] = numpy.maximum(E2.low[idx], ridges.low)
            E3.low[idx] = numpy.maximum(E3.low[idx], ridges.low)
            E4.low[idx] = numpy.maximum(E4.low[idx], ridges.low)
        else:
            E1[idx] = max_Stats(E1[idx], ridges)
            E2[idx] = max_Stats(E2[idx], ridges)
            E3[idx] = max_Stats(E3[idx], ridges)
            E4[idx] = max_Stats(E4[idx], ridges)

    def fold_interior_ridge_equal(self, dir, idx, ridges):
        """Fold out ridge for equal cases"""
        self.fold_interior_ridge(dir, idx, ridges)
        self.fold_interior_ridge(od(dir), idx, ridges)

    def find_deepest_exterior_corner(self, dir, matlab=False, adjust_centers=True):
        # Exterior corner of interest
        ec_u = self.sec(dir[1]+dir)
        ec_v = self.sec(dir[0]+dir)

        # The following four edges should be identical!
        # Interior corner
        ic_u = self.sec(dir[1])
        ic_v = self.sec(dir[0])

        # opposing interior corner
        ic_opu = self.sec(od(dir[1]))
        ic_opv = self.sec(od(dir[0]))

        # opposing exterior corner
        ec_opu = self.sec(od(dir[1])+od(dir[0])+od(dir[1]))
        ec_opv = self.sec(od(dir[0])+od(dir[0])+od(dir[1]))

        # neighor exterior corner in the zonal direction
        ec_nzu = self.sec(od(dir[1])+dir[0]+od(dir[1]))
        ec_nzv = self.sec(dir[0]+dir[0]+od(dir[1]))

        # neighor exterior corner in the meridional direction
        ec_nmu = self.sec(dir[1]+od(dir[0])+dir[1])
        ec_nmv = self.sec(od(dir[0])+od(dir[0])+dir[1])

        crnr_ex = numpy.maximum(ec_u.low, ec_v.low)
        crnr_ex_op = numpy.minimum(numpy.maximum(ec_opu.low, ec_opv.low),
                                   numpy.minimum(numpy.maximum(ec_nzu.low, ec_nzv.low),
                                                 numpy.maximum(ec_nmu.low, ec_nmv.low)))
        crnr_in = numpy.minimum(ic_u.low, ic_v.low)
        crnr_in_nz = numpy.maximum(ic_u.low, ic_opv.low)
        crnr_in_nm = numpy.maximum(ic_v.low, ic_opu.low)

        idx = numpy.nonzero( (crnr_ex < crnr_ex_op) & (crnr_ex < crnr_in) )

        inner = StatsBase(low=numpy.minimum(crnr_in_nz, crnr_in_nm),
                          ave=0.5*(crnr_in_nz+crnr_in_nm),
                          hgh=numpy.maximum(crnr_in_nm, crnr_in_nm))

        if matlab:
            adjust_centers = False
        # adjust inner edges
        if matlab:
            ic_u.low[idx] = crnr_ex[idx]
            ic_v.low[idx] = crnr_ex[idx]
            ic_opu.low[idx] = crnr_ex[idx]
            ic_opv.low[idx] = crnr_ex[idx]
        else:
            ic_u.low[idx] = numpy.minimum(ic_u.low[idx], crnr_ex[idx])
            ic_v.low[idx] = numpy.minimum(ic_v.low[idx], crnr_ex[idx])
            ic_opu.low[idx] = numpy.minimum(ic_opu.low[idx], crnr_ex[idx])
            ic_opv.low[idx] = numpy.minimum(ic_opv.low[idx], crnr_ex[idx])

        if adjust_centers:
            for dir in ['SW', 'SE', 'NW', 'NE']:
                self.sec(dir).low[idx] = numpy.minimum(self.sec(dir).low[idx], crnr_ex[idx])
        return idx, inner[idx]

    def expand_interior_corner(self, dir, idx, inners, matlab=False):
        # opposing exterior corner
        ec_opu = self.sec(od(dir[1])+od(dir[0])+od(dir[1]))
        ec_opv = self.sec(od(dir[0])+od(dir[0])+od(dir[1]))

        # neighor exterior corner in the zonal direction
        ec_nzu = self.sec(od(dir[1])+dir[0]+od(dir[1]))
        ec_nzv = self.sec(dir[0]+dir[0]+od(dir[1]))

        # neighor exterior corner in the meridional direction
        ec_nmu = self.sec(dir[1]+od(dir[0])+dir[1])
        ec_nmv = self.sec(od(dir[0])+od(dir[0])+dir[1])

        if matlab:
            ec_opu[idx] = max_Stats(ec_opu[idx], inners)
            ec_opv[idx] = max_Stats(ec_opv[idx], inners)
            ec_nzu[idx] = max_Stats(ec_nzu[idx], inners)
            ec_nzv[idx] = max_Stats(ec_nzv[idx], inners)
            ec_nmu[idx] = max_Stats(ec_nmu[idx], inners)
            ec_nmv[idx] = max_Stats(ec_nmv[idx], inners)
        else:
            ec_opu.low[idx] = numpy.maximum(ec_opu.low[idx], inners.low)
            ec_opv.low[idx] = numpy.maximum(ec_opv.low[idx], inners.low)
            ec_nzu.low[idx] = numpy.maximum(ec_nzu.low[idx], inners.low)
            ec_nzv.low[idx] = numpy.maximum(ec_nzv.low[idx], inners.low)
            ec_nmu.low[idx] = numpy.maximum(ec_nmu.low[idx], inners.low)
            ec_nmv.low[idx] = numpy.maximum(ec_nmv.low[idx], inners.low)

    def expand_interior_corners(self, adjust_centers=False, matlab=False, verbose=False):
        # Check if all inner edges are equal (they should be at this step)
        s, n, w, e = self.sec('S').low, self.sec('N').low, self.sec('W').low, self.sec('E').low
        assert numpy.all(s==n) and numpy.all(w==e) and numpy.all(s==w), 'Not all inner edges are equal before exterior corner.'

        # Preserve the deepest exterior corner
        idx_sw, corner_sw = self.find_deepest_exterior_corner('SW', adjust_centers=adjust_centers, matlab=matlab)
        idx_se, corner_se = self.find_deepest_exterior_corner('SE', adjust_centers=adjust_centers, matlab=matlab)
        idx_nw, corner_nw = self.find_deepest_exterior_corner('NW', adjust_centers=adjust_centers, matlab=matlab)
        idx_ne, corner_ne = self.find_deepest_exterior_corner('NE', adjust_centers=adjust_centers, matlab=matlab)

        if verbose:
            print("  SW: {}".format(idx_sw[0].size))
            print("  NW: {}".format(idx_se[0].size))
            print("  SE: {}".format(idx_nw[0].size))
            print("  NE: {}".format(idx_ne[0].size))

        self.expand_interior_corner('SW', idx_sw, corner_sw, matlab=matlab)
        self.expand_interior_corner('SE', idx_se, corner_se, matlab=matlab)
        self.expand_interior_corner('NW', idx_nw, corner_nw, matlab=matlab)
        self.expand_interior_corner('NE', idx_ne, corner_ne, matlab=matlab)

    def limit_connections(self, connections={}, verbose=False):
        if verbose:
            print('limit connections')
        idx = dict().fromkeys(connections.keys())

        # find the indices
        for path, sill in connections.items():
            d1, d2 = path
            sec1a, sec1b = exterior_edges(d1)
            sec2a, sec2b = exterior_edges(d2)
            w1 = numpy.minimum( self.sec(sec1a).low, self.sec(sec1b).low )
            w2 = numpy.minimum( self.sec(sec2a).low, self.sec(sec2b).low )

            i1 = numpy.nonzero((sill>w1) & (w1>=w2))
            i2 = numpy.nonzero((sill>w2) & (w1<=w2))
            if verbose:
                print("  {:s} wall raised from {:s} pathway: {:d}".format(d1, path, i1[0].size))
                print("  {:s} wall raised from {:s} pathway: {:d}".format(d2, path, i2[0].size))
            idx[path] = (i1, i2)

        # adjust edges
        for path, sill in connections.items():
            d1, d2 = path
            sec1a, sec1b = exterior_edges(d1)
            sec2a, sec2b = exterior_edges(d2)

            i1, i2 = idx[path]
            self.sec(sec1a).low[i1] = numpy.maximum(self.sec(sec1a).low[i1], sill[i1])
            self.sec(sec1b).low[i1] = numpy.maximum(self.sec(sec1b).low[i1], sill[i1])

            self.sec(sec2a).low[i2] = numpy.maximum(self.sec(sec2a).low[i2], sill[i2])
            self.sec(sec2b).low[i2] = numpy.maximum(self.sec(sec2b).low[i2], sill[i2])

    def push_corners(self, update_interior_mean_max=True, matlab=False, verbose=False):
        """Folds out tallest corners. Acts only on "effective" values.

        A convex corner within a coarse grid cell can be made into a
        concave corner without changing connectivity across the major
        parts of the cell. The cross-corner connection for the minor
        part of the cell is eliminated."""

        if verbose: print("Begin push_corners")
        if verbose: print("  SW: ", end="")
        self.push_corners_sw(update_interior_mean_max=update_interior_mean_max, matlab=matlab, verbose=verbose) # Push SW
        # Alias
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        # Flip in j direction
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        if verbose: print("  NW: ", end="")
        self.push_corners_sw(update_interior_mean_max=update_interior_mean_max, matlab=matlab, verbose=verbose) # Push NW
        # Flip in i direction
        C.flip(axis=1)
        U.flip(axis=1)
        V.flip(axis=1)
        if verbose: print("  NE: ", end="")
        self.push_corners_sw(update_interior_mean_max=update_interior_mean_max, matlab=matlab, verbose=verbose) # Push NE
        # Flip in j direction
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        if verbose: print("  SE: ", end="")
        self.push_corners_sw(update_interior_mean_max=update_interior_mean_max, matlab=matlab, verbose=verbose) # Push SE
        # Flip in i direction
        C.flip(axis=1)
        U.flip(axis=1)
        V.flip(axis=1)
    def push_corners_sw(self, update_interior_mean_max=True, matlab=True, verbose=False):
        """Folds out SW corner is it is the highest ridge. Acts only on "effective" values.

        A convex corner within a coarse grid cell can be made into a
        concave corner without changing connectivity across the major
        parts of the cell. The cross-corner connection for the minor
        part of the cell is eliminated."""
        # Alias
        C,U,V = self.c_effective,self.u_effective,self.v_effective
        # Inner SW corner
        crnr_min = numpy.minimum( U.low[::2,1::2], V.low[1::2,::2] )    # Min or "sill" for SW corner
        crnr_mean = 0.5*( U.ave[::2,1::2] + V.ave[1::2,::2] )         # Mean for SW corner
        crnr_max = numpy.maximum( U.hgh[::2,1::2], V.hgh[1::2,::2] )    # Max for SW corner
        # Values for the coarse cell outside of the SW corner
        opp_ridge = numpy.maximum( U.low[1::2,1::2], V.low[1::2,1::2] ) # Ridge for NE corner
        opp_cmean = ( ( C.ave[::2,1::2] + C.ave[1::2,::2] ) + C.ave[1::2,1::2] )/3 # Mean of outer cells
        j,i = numpy.nonzero( crnr_min>opp_ridge )  # Find where the SW corner has the highest sill
        if len(i)>0:
            J,I = 2*j,2*i
            # Replace inner minimum values with ridge value
            # - set inner SW corner sill to peak of the NW ridge to avoid introducing a new deep diagonal
            #   connection across the interior of the coarse cell
            U.low[J,I+1] = opp_ridge[j,i]
            V.low[J+1,I] = opp_ridge[j,i]
            # ????? No replace inner mean and max ???? Not used?
            # Override outer SW edge values with SW corner inner values
            U.low[J,I] = numpy.maximum( U.low[J,I], crnr_min[j,i] )
            V.low[J,I] = numpy.maximum( V.low[J,I], crnr_min[j,i] )
            U.ave[J,I] = numpy.maximum( U.ave[J,I], crnr_mean[j,i] )
            V.ave[J,I] = numpy.maximum( V.ave[J,I], crnr_mean[j,i] )
            U.hgh[J,I] = numpy.maximum( U.hgh[J,I], crnr_max[j,i] )
            V.hgh[J,I] = numpy.maximum( V.hgh[J,I], crnr_max[j,i] )
            # Override SW cell values with outer values from coarse cell
            C.low[J,I] = opp_ridge[j,i] # This will be taller than other minimums but is it lower than opp_cmean ????
            if matlab:
                C.ave[J,I] = opp_cmean[j,i]
                C.hgh[J,I] = opp_ridge[j,i]
                update_interior_mean_max = False
            if update_interior_mean_max:
                C.ave[J,I] = numpy.maximum( C.ave[J,I], opp_cmean[j,i] ) # Avoids changing the mean of the remaining coarse cell
                C.hgh[J,I] = numpy.maximum( C.hgh[J,I], opp_ridge[j,i] )   # Will be taller than cell means?
                #opp_ridge = 0.5*( U.ave[1::2,1::2] + V.ave[1::2,1::2] ) # Ridge for NE corner
                U.ave[J,I+1] = opp_ridge[j,i]
                V.ave[J+1,I] = opp_ridge[j,i]
                #opp_ridge = numpy.maximum( U.hgh[1::2,1::2], V.hgh[1::2,1::2] ) # Ridge for NE corner
                U.hgh[J,I+1] = opp_ridge[j,i]
                V.hgh[J+1,I] = opp_ridge[j,i]
        if verbose: print(j.size, " pushed")
    def lower_tallest_buttress(self, update_interior_mean=True, verbose=False):
        """Lower tallest barrier to remove buttress"""
        if verbose: print("Begin lower_tallest_buttress")
        # Alias lowest
        C,U,V = self.c_effective.low,self.u_effective.low,self.v_effective.low
        # Find where the S ridge is higher than other 3
        oppo3 = numpy.maximum( U[1::2,1::2], numpy.maximum( V[1::2,::2], V[1::2,1::2] ) )
        j,i = numpy.nonzero( U[::2,1::2]>oppo3 )
        U[2*j,2*i+1] = oppo3[j,i]
        if verbose: print("  S ridge (low): ", j.size, ' removed')
        # Find where the N ridge is higher than other 3
        oppo3 = numpy.maximum( U[::2,1::2], numpy.maximum( V[1::2,::2], V[1::2,1::2] ) )
        j,i = numpy.nonzero( U[1::2,1::2]>oppo3 )
        U[2*j+1,2*i+1] = oppo3[j,i]
        if verbose: print("  N ridge (low): ", j.size, ' removed')
        # Find where the W ridge is higher than other 3
        oppo3 = numpy.maximum( V[1::2,1::2], numpy.maximum( U[::2,1::2], U[1::2,1::2] ) )
        j,i = numpy.nonzero( V[1::2,::2]>oppo3 )
        V[2*j+1,2*i] = oppo3[j,i]
        if verbose: print("  W ridge (low): ", j.size, ' removed')
        # Find where the E ridge is higher than other 3
        oppo3 = numpy.maximum( V[1::2,::2], numpy.maximum( U[::2,1::2], U[1::2,1::2] ) )
        j,i = numpy.nonzero( V[1::2,1::2]>oppo3 )
        V[2*j+1,2*i+1] = oppo3[j,i]
        if verbose: print("  E ridge (low): ", j.size, ' removed')

        # Alias for averages
        if update_interior_mean:
            C,U,V = self.c_effective.ave,self.u_effective.ave,self.v_effective.ave
            # Find where the S ridge is higher than other 3
            oppo3 = numpy.maximum( U[1::2,1::2], numpy.maximum( V[1::2,::2], V[1::2,1::2] ) )
            j,i = numpy.nonzero( U[::2,1::2]>oppo3 )
            U[2*j,2*i+1] = oppo3[j,i]
            if verbose: print("  S ridge (ave): ", j.size, ' removed')
            # Find where the N ridge is higher than other 3
            oppo3 = numpy.maximum( U[::2,1::2], numpy.maximum( V[1::2,::2], V[1::2,1::2] ) )
            j,i = numpy.nonzero( U[1::2,1::2]>oppo3 )
            U[2*j+1,2*i+1] = oppo3[j,i]
            if verbose: print("  N ridge (ave): ", j.size, ' removed')
            # Find where the W ridge is higher than other 3
            oppo3 = numpy.maximum( V[1::2,1::2], numpy.maximum( U[::2,1::2], U[1::2,1::2] ) )
            j,i = numpy.nonzero( V[1::2,::2]>oppo3 )
            V[2*j+1,2*i] = oppo3[j,i]
            if verbose: print("  W ridge (ave): ", j.size, ' removed')
            # Find where the E ridge is higher than other 3
            oppo3 = numpy.maximum( V[1::2,::2], numpy.maximum( U[::2,1::2], U[1::2,1::2] ) )
            j,i = numpy.nonzero( V[1::2,1::2]>oppo3 )
            V[2*j+1,2*i+1] = oppo3[j,i]
            if verbose: print("  E ridge (ave): ", j.size, ' removed')
    def fold_out_central_ridges(self, matlab=False, verbose=False):
        """Folded out interior ridges to the sides of the coarse cell"""
        if verbose: print("Begin fold_out_central_ridges")
        if verbose: print("  S: ", end="")
        self.fold_out_central_ridge_s(matlab=matlab, verbose=verbose)
        if matlab:
            if verbose: print("  S=N: ", end="")
            self.fold_out_central_ridge_ns(verbose=verbose)
        # Alias
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        # Flip in j direction so j=S, i=E
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        if verbose: print("  N: ", end="")
        self.fold_out_central_ridge_s(matlab=matlab, verbose=verbose)
        # Transpose so j=E, i=S
        C.transpose()
        U.transpose()
        V.transpose()
        self.u_effective,self.v_effective = self.v_effective,self.u_effective
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        if verbose: print("  W: ", end="")
        self.fold_out_central_ridge_s(matlab=matlab, verbose=verbose)
        if matlab:
            if verbose: print("  W=E: ", end="")
            self.fold_out_central_ridge_ns(verbose=verbose)
        # Flip in j direction so j=W, i=S
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        if verbose: print("  E: ", end="")
        self.fold_out_central_ridge_s(matlab=matlab, verbose=verbose)
        # Undo transformations
        C.transpose()
        U.transpose()
        V.transpose()
        self.u_effective,self.v_effective = self.v_effective,self.u_effective
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        C.flip(axis=1)
        U.flip(axis=1)
        V.flip(axis=1)
    def fold_out_central_ridge_s(self, matlab=True, verbose=False):
        """An interior east-west ridge is folded out to the southern outer edges if it
        is the tallest central ridge and the south is the taller half to expand to."""
        # Alias
        C,U,V = self.c_effective,self.u_effective,self.v_effective
        ew_ridge_low = numpy.minimum( V.low[1::2,::2], V.low[1::2,1::2] )
        #ew_ridge_hgh = numpy.maximum( V.hgh[1::2,::2], V.hgh[1::2,1::2] )
        #ew_ridge_ave = 0.5*( V.low[1::2,::2] + V.low[1::2,1::2] )
        if matlab:
            ew_ridge_hgh = numpy.maximum( V.hgh[1::2,::2], V.hgh[1::2,1::2] )
            ew_ridge_ave = 0.5*( V.ave[1::2,::2] + V.ave[1::2,1::2] )
        ns_ridge_low_min = numpy.minimum( U.low[::2,1::2], U.low[1::2,1::2] )
        ns_ridge_low_max = numpy.maximum( U.low[::2,1::2], U.low[1::2,1::2] )
        # Coarse cell index j,i
        j,i = numpy.nonzero(
              ( ( ew_ridge_low>ns_ridge_low_min) & (ew_ridge_low>=ns_ridge_low_max ) ) # E-W ridge is the taller ridge
              & (
                  ( U.low[::2,1::2] > U.low[1::2,1::2] ) # Southern buttress is taller than north
                  | (
                      ( U.low[::2,1::2] >= U.low[1::2,1::2] ) # Southern buttress is equal to the north
                      & (
                          ( C.low[::2,::2]+C.low[::2,1::2] > C.low[1::2,::2]+C.low[1::2,1::2] ) | # Southern cells are higher than north on average
                          ( V.low[:-1:2,::2]+V.low[:-1:2,1::2] > V.low[2::2,::2]+V.low[2::2,1::2] ) # Southern edges are higher than north on average
                ) ) ) )

        # # E-W ridge is the taller ridge
        # ew_ridges = ( ( ew_ridge_low>ns_ridge_low_min) & (ew_ridge_low>=ns_ridge_low_max ) )
        # high_buttress = ( U.low[::2,1::2]>U.low[1::2,1::2] ) # Southern buttress is taller than north
        # high_cell = (  ( U.low[::2,1::2]==U.low[1::2,1::2] )
        #              & ( C.low[::2,::2]+C.low[::2,1::2]>C.low[1::2,::2]+C.low[1::2,1::2] ) )  # Southern buttress is equal to the north
        # high_edge = (  ( U.low[::2,1::2]==U.low[1::2,1::2] )
        #              & ( C.low[::2,::2]+C.low[::2,1::2]==C.low[1::2,::2]+C.low[1::2,1::2] )
        #              & ( V.low[:-1:2,::2]+V.low[:-1:2,1::2]>V.low[2::2,::2]+V.low[2::2,1::2]) )
        # j,i = numpy.nonzero( ew_ridges & (high_buttress | high_cell | high_edge) )

        J,I = 2*j,2*i
        # Outer edges of southern half
        U.low[J,I] = numpy.maximum( U.low[J,I], ew_ridge_low[j,i] )
        V.low[J,I] = numpy.maximum( V.low[J,I], ew_ridge_low[j,i] )
        V.low[J,I+1] = numpy.maximum( V.low[J,I+1], ew_ridge_low[j,i] )
        U.low[J,I+2] = numpy.maximum( U.low[J,I+2], ew_ridge_low[j,i] )
        if matlab:
            U.ave[J,I] = numpy.maximum( U.ave[J,I], ew_ridge_ave[j,i] )
            V.ave[J,I] = numpy.maximum( V.ave[J,I], ew_ridge_ave[j,i] )
            V.ave[J,I+1] = numpy.maximum( V.ave[J,I+1], ew_ridge_ave[j,i] )
            U.ave[J,I+2] = numpy.maximum( U.ave[J,I+2], ew_ridge_ave[j,i] )
            U.hgh[J,I] = numpy.maximum( U.hgh[J,I], ew_ridge_hgh[j,i] )
            V.hgh[J,I] = numpy.maximum( V.hgh[J,I], ew_ridge_hgh[j,i] )
            V.hgh[J,I+1] = numpy.maximum( V.hgh[J,I+1], ew_ridge_hgh[j,i] )
            U.hgh[J,I+2] = numpy.maximum( U.hgh[J,I+2], ew_ridge_hgh[j,i] )
        # Replace E-W ridge
        V.low[J+1,I] = ns_ridge_low_min[j,i]
        V.low[J+1,I+1] = ns_ridge_low_min[j,i]
        # E-W ridge hgh and ave not modified??
        # Southern cells
        C.low[J,I] = ns_ridge_low_min[j,i]
        C.low[J,I+1] = ns_ridge_low_min[j,i]
        U.low[J,I+1] = ns_ridge_low_min[j,i]

        if matlab:
            C.ave[J,I] = 0.5*( C.ave[J+1,I] + C.ave[J+1,I+1] )
            C.ave[J,I+1] = 0.5*( C.ave[J+1,I] + C.ave[J+1,I+1] )
            C.hgh[J,I] = ns_ridge_low_min[j,i]
            C.hgh[J,I+1] = ns_ridge_low_min[j,i]

        if verbose: print(j.size, " folded")
    def fold_out_central_ridge_ns(self, verbose=False):
        """An interior east-west ridge is folded out to the southern outer edges if it
        is the tallest central ridge and the south is the taller half to expand to."""
        # Alias
        C,U,V = self.c_effective,self.u_effective,self.v_effective
        ew_ridge_low = numpy.minimum( V.low[1::2,::2], V.low[1::2,1::2] )
        #ew_ridge_hgh = numpy.maximum( V.hgh[1::2,::2], V.hgh[1::2,1::2] )
        #ew_ridge_ave = 0.5*( V.low[1::2,::2] + V.low[1::2,1::2] )
        ns_ridge_low_min = numpy.minimum( U.low[::2,1::2], U.low[1::2,1::2] )
        ns_ridge_low_max = numpy.maximum( U.low[::2,1::2], U.low[1::2,1::2] )
        # Coarse cell index j,i
        j,i = numpy.nonzero(
              (  ( ew_ridge_low>ns_ridge_low_min) & (ew_ridge_low>=ns_ridge_low_max ) ) # E-W ridge is the taller ridge
            & (  ( U.low[::2,1::2] == U.low[1::2,1::2] ) # Southern buttress is equal to the north
               & ( C.low[::2,::2]+C.low[::2,1::2] == C.low[1::2,::2]+C.low[1::2,1::2] )  # Southern cells are equal to north on average
               & ( V.low[:-1:2,::2]+V.low[:-1:2,1::2] == V.low[2::2,::2]+V.low[2::2,1::2] ) # Southern edges are equal to north on average
              ) )
        J,I = 2*j,2*i

        # Old by HW
        # # Outer edges of southern half
        # U.low[J,I] = numpy.maximum( U.low[J,I], ew_ridge_low[j,i] )
        # V.low[J,I] = numpy.maximum( V.low[J,I], ew_ridge_low[j,i] )
        # V.low[J,I+1] = numpy.maximum( V.low[J,I+1], ew_ridge_low[j,i] )
        # U.low[J,I+2] = numpy.maximum( U.low[J,I+2], ew_ridge_low[j,i] )

        # # Outer edges of northern half
        # U.low[J+1,I] = numpy.maximum( U.low[J+1,I], ew_ridge_low[j,i] )
        # V.low[J+2,I] = numpy.maximum( V.low[J+2,I], ew_ridge_low[j,i] )
        # V.low[J+2,I+1] = numpy.maximum( V.low[J+2,I+1], ew_ridge_low[j,i] )
        # U.low[J+1,I+2] = numpy.maximum( U.low[J+1,I+2], ew_ridge_low[j,i] )

        # # Replace E-W ridge
        # V.low[J+1,I] = ns_ridge_low_min[j,i]
        # V.low[J+1,I+1] = ns_ridge_low_min[j,i]
        # # Southern cells
        # C.low[J,I] = ns_ridge_low_min[j,i]
        # C.low[J,I+1] = ns_ridge_low_min[j,i]
        # U.low[J,I+1] = ns_ridge_low_min[j,i]
        # # Northern cells
        # C.low[J+1,I] = ns_ridge_low_min[j,i]
        # C.low[J+1,I+1] = ns_ridge_low_min[j,i]
        # U.low[J+1,I+1] = ns_ridge_low_min[j,i]

        # MatLab
        # Outer edges of southern half
        U.low[J,I] = numpy.maximum( U.low[J,I], ew_ridge_low[j,i] )
        U.low[J,I+2] = numpy.maximum( U.low[J,I+2], ew_ridge_low[j,i] )

        V.low[J,I] = numpy.maximum( V.low[J,I], ew_ridge_low[j,i] )
        V.low[J,I+1] = numpy.maximum( V.low[J,I+1], ew_ridge_low[j,i] )

        # Outer edges of northern half
        U.low[J+1,I] = numpy.maximum( U.low[J+1,I], ew_ridge_low[j,i] )
        U.low[J+1,I+2] = numpy.maximum( U.low[J+1,I+2], ew_ridge_low[j,i] )

        V.low[J+2,I] = numpy.maximum( V.low[J+2,I], ew_ridge_low[j,i] )
        V.low[J+2,I+1] = numpy.maximum( V.low[J+2,I+1], ew_ridge_low[j,i] )

        U.low[J,I+1] = ew_ridge_low[j,i]
        U.low[J+1,I+1] = ew_ridge_low[j,i]
        V.low[J+1,I] = ew_ridge_low[j,i]
        V.low[J+1,I+1] = ew_ridge_low[j,i]

        # MatLab does this, don't think it works
        # C.low[J,I] = ew_ridge_low[j,i] * numpy.nan
        # C.low[J,I+1] = ew_ridge_low[j,i] * numpy.nan
        # C.low[J+1,I] = ew_ridge_low[j,i] * numpy.nan
        # C.low[J+1,I+1] = ew_ridge_low[j,i] * numpy.nan
        C.low[J,I] = ew_ridge_low[j,i]
        C.low[J,I+1] = ew_ridge_low[j,i]
        C.low[J+1,I] = ew_ridge_low[j,i]
        C.low[J+1,I+1] = ew_ridge_low[j,i]

        if verbose: print(j.size, " folded")
    def invert_exterior_corners(self, matlab=False, verbose=False):
        """The deepest exterior corner is expanded to fill the coarse cell"""
        if verbose: print("Begin invert_exterior_corners")
        # Alias
        C,U,V = self.c_effective,self.u_effective,self.v_effective
        # Exterior deep corners
        d_sw = numpy.maximum( U.low[::2,:-1:2], V.low[:-1:2,::2] )
        d_se = numpy.maximum( U.low[::2,2::2], V.low[:-1:2,1::2] )
        d_nw = numpy.maximum( U.low[1::2,:-1:2], V.low[2::2,::2] )
        d_ne = numpy.maximum( U.low[1::2,2::2], V.low[2::2,1::2] )
        # Interior sills
        s_sw = numpy.minimum( U.low[::2,1::2], V.low[1::2,::2] )
        s_se = numpy.minimum( U.low[::2,1::2], V.low[1::2,1::2] )
        s_nw = numpy.minimum( U.low[1::2,1::2], V.low[1::2,::2] )
        s_ne = numpy.minimum( U.low[1::2,1::2], V.low[1::2,1::2] )
        # Diagonal ridges from corners
        r_sw = numpy.maximum( U.low[::2,1::2], V.low[1::2,::2] )
        r_se = numpy.maximum( U.low[::2,1::2], V.low[1::2,1::2] )
        r_nw = numpy.maximum( U.low[1::2,1::2], V.low[1::2,::2] )
        r_ne = numpy.maximum( U.low[1::2,1::2], V.low[1::2,1::2] )

        # SW conditions
        oppo = numpy.minimum( d_ne, numpy.minimum( d_nw, d_se ) )
        swj,swi = numpy.nonzero( (d_sw < oppo) & (d_sw < s_sw) ) # SW is deepest corner

        # SE conditions
        oppo = numpy.minimum( d_nw, numpy.minimum( d_ne, d_sw ) )
        sej,sei = numpy.nonzero( (d_se < oppo) & (d_se < s_se) ) # SE is deepest corner

        # NE conditions
        oppo = numpy.minimum( d_sw, numpy.minimum( d_se, d_nw ) )
        nej,nei = numpy.nonzero( (d_ne < oppo) & (d_ne < s_ne) ) # NE is deepest corner

        # NW conditions
        oppo = numpy.minimum( d_se, numpy.minimum( d_sw, d_ne ) )
        nwj,nwi = numpy.nonzero( (d_nw < oppo) & (d_nw < s_nw) ) # NW is deepest corner

        # Apply SW
        j,i,J,I=swj,swi,2*swj,2*swi
        # Deepen interior walls and cells
        if matlab:
            U.low[J,I+1] = d_sw[j,i]
            U.low[J+1,I+1] = d_sw[j,i]
            V.low[J+1,I] = d_sw[j,i]
            V.low[J+1,I+1] = d_sw[j,i]
            # C is not treated in MatLab
        else: # minimum is not necessary as all interior wall low are of the same height and would be higher than d_sw here
            U.low[J,I+1] = numpy.minimum( U.low[J,I+1], d_sw[j,i] )
            U.low[J+1,I+1] = numpy.minimum( U.low[J+1,I+1], d_sw[j,i] )
            V.low[J+1,I] = numpy.minimum( V.low[J+1,I], d_sw[j,i] )
            V.low[J+1,I+1] = numpy.minimum( V.low[J+1,I+1], d_sw[j,i] )
            C.low[J,I] = numpy.minimum( C.low[J,I], d_sw[j,i] )
            C.low[J,I+1] = numpy.minimum( C.low[J,I+1], d_sw[j,i] )
            C.low[J+1,I] = numpy.minimum( C.low[J+1,I], d_sw[j,i] )
            C.low[J+1,I+1] = numpy.minimum( C.low[J+1,I+1], d_sw[j,i] )
        # Outer edges
        if matlab:
            new_ridge = numpy.minimum( r_se, r_nw )
            V.low[J,I+1] = numpy.maximum( V.low[J,I+1], new_ridge[j,i] )
            U.low[J,I+2] = numpy.maximum( U.low[J,I+2], new_ridge[j,i] )
            U.low[J+1,I+2] = numpy.maximum( U.low[J+1,I+2], new_ridge[j,i] )
            V.low[J+2,I+1] = numpy.maximum( V.low[J+2,I+1], new_ridge[j,i] )
            V.low[J+2,I] = numpy.maximum( V.low[J+2,I], new_ridge[j,i] )
            U.low[J+1,I] = numpy.maximum( U.low[J+1,I], new_ridge[j,i] )
            new_ridge = 0.5 * ( r_se + r_nw )
            V.ave[J,I+1] = numpy.maximum( V.ave[J,I+1], new_ridge[j,i] )
            U.ave[J,I+2] = numpy.maximum( U.ave[J,I+2], new_ridge[j,i] )
            U.ave[J+1,I+2] = numpy.maximum( U.ave[J+1,I+2], new_ridge[j,i] )
            V.ave[J+2,I+1] = numpy.maximum( V.ave[J+2,I+1], new_ridge[j,i] )
            V.ave[J+2,I] = numpy.maximum( V.ave[J+2,I], new_ridge[j,i] )
            U.ave[J+1,I] = numpy.maximum( U.ave[J+1,I], new_ridge[j,i] )
            new_ridge = numpy.maximum( r_se, r_nw )
            V.hgh[J,I+1] = numpy.maximum( V.hgh[J,I+1], new_ridge[j,i] )
            U.hgh[J,I+2] = numpy.maximum( U.hgh[J,I+2], new_ridge[j,i] )
            U.hgh[J+1,I+2] = numpy.maximum( U.hgh[J+1,I+2], new_ridge[j,i] )
            V.hgh[J+2,I+1] = numpy.maximum( V.hgh[J+2,I+1], new_ridge[j,i] )
            V.hgh[J+2,I] = numpy.maximum( V.hgh[J+2,I], new_ridge[j,i] )
            U.hgh[J+1,I] = numpy.maximum( U.hgh[J+1,I], new_ridge[j,i] )
        else:
            new_ridge = numpy.minimum( r_se, r_nw )
            V.low[J,I+1] = numpy.maximum( V.low[J,I+1], r_se[j,i] )
            U.low[J,I+2] = numpy.maximum( U.low[J,I+2], r_se[j,i] )
            U.low[J+1,I+2] = numpy.maximum( U.low[J+1,I+2], new_ridge[j,i] )
            V.low[J+2,I+1] = numpy.maximum( V.low[J+2,I+1], new_ridge[j,i] )
            V.low[J+2,I] = numpy.maximum( V.low[J+2,I], r_nw[j,i] )
            U.low[J+1,I] = numpy.maximum( U.low[J+1,I], r_nw[j,i] )

        if verbose: print("  SW: ", swj.size, " inverted")

        # Apply SE
        j,i,J,I=sej,sei,2*sej,2*sei
        # Deepen interior walls and cells
        if matlab:
            U.low[J,I+1] = d_se[j,i]
            U.low[J+1,I+1] = d_se[j,i]
            V.low[J+1,I] = d_se[j,i]
            V.low[J+1,I+1] = d_se[j,i]
        else:
            U.low[J,I+1] = numpy.minimum( U.low[J,I+1], d_se[j,i] )
            U.low[J+1,I+1] = numpy.minimum( U.low[J+1,I+1], d_se[j,i] )
            V.low[J+1,I] = numpy.minimum( V.low[J+1,I], d_se[j,i] )
            V.low[J+1,I+1] = numpy.minimum( V.low[J+1,I+1], d_se[j,i] )
            C.low[J,I] = numpy.minimum( C.low[J,I], d_se[j,i] )
            C.low[J,I+1] = numpy.minimum( C.low[J,I+1], d_se[j,i] )
            C.low[J+1,I] = numpy.minimum( C.low[J+1,I], d_se[j,i] )
            C.low[J+1,I+1] = numpy.minimum( C.low[J+1,I+1], d_se[j,i] )
        # Outer edges
        if matlab:
            new_ridge = numpy.minimum( r_sw, r_ne )
            V.low[J,I] = numpy.maximum( V.low[J,I], new_ridge[j,i] )
            U.low[J,I] = numpy.maximum( U.low[J,I], new_ridge[j,i] )
            U.low[J+1,I] = numpy.maximum( U.low[J+1,I], new_ridge[j,i] )
            V.low[J+2,I] = numpy.maximum( V.low[J+2,I], new_ridge[j,i] )
            V.low[J+2,I+1] = numpy.maximum( V.low[J+2,I+1], new_ridge[j,i] )
            U.low[J+1,I+2] = numpy.maximum( U.low[J+1,I+2], new_ridge[j,i] )
            new_ridge = 0.5 * ( r_sw + r_ne )
            V.ave[J,I] = numpy.maximum( V.ave[J,I], new_ridge[j,i] )
            U.ave[J,I] = numpy.maximum( U.ave[J,I], new_ridge[j,i] )
            U.ave[J+1,I] = numpy.maximum( U.ave[J+1,I], new_ridge[j,i] )
            V.ave[J+2,I] = numpy.maximum( V.ave[J+2,I], new_ridge[j,i] )
            V.ave[J+2,I+1] = numpy.maximum( V.ave[J+2,I+1], new_ridge[j,i] )
            U.ave[J+1,I+2] = numpy.maximum( U.ave[J+1,I+2], new_ridge[j,i] )
            new_ridge = numpy.maximum( r_sw, r_ne )
            V.hgh[J,I] = numpy.maximum( V.hgh[J,I], new_ridge[j,i] )
            U.hgh[J,I] = numpy.maximum( U.hgh[J,I], new_ridge[j,i] )
            U.hgh[J+1,I] = numpy.maximum( U.hgh[J+1,I], new_ridge[j,i] )
            V.hgh[J+2,I] = numpy.maximum( V.hgh[J+2,I], new_ridge[j,i] )
            V.hgh[J+2,I+1] = numpy.maximum( V.hgh[J+2,I+1], new_ridge[j,i] )
            U.hgh[J+1,I+2] = numpy.maximum( U.hgh[J+1,I+2], new_ridge[j,i] )
        else:
            new_ridge = numpy.minimum( r_sw, r_ne )
            V.low[J,I] = numpy.maximum( V.low[J,I], r_sw[j,i] )
            U.low[J,I] = numpy.maximum( U.low[J,I], r_sw[j,i] )
            U.low[J+1,I] = numpy.maximum( U.low[J+1,I], new_ridge[j,i] )
            V.low[J+2,I] = numpy.maximum( V.low[J+2,I], new_ridge[j,i] )
            V.low[J+2,I+1] = numpy.maximum( V.low[J+2,I+1], r_ne[j,i] )
            U.low[J+1,I+2] = numpy.maximum( U.low[J+1,I+2], r_ne[j,i] )
        if verbose: print("  SE: ", sej.size, " inverted")

        # Apply NW
        j,i,J,I=nwj,nwi,2*nwj,2*nwi
        # Deepen interior walls and cells
        if matlab:
            U.low[J,I+1] = d_nw[j,i]
            U.low[J+1,I+1] = d_nw[j,i]
            V.low[J+1,I] = d_nw[j,i]
            V.low[J+1,I+1] = d_nw[j,i]
        else:
            U.low[J,I+1] = numpy.minimum( U.low[J,I+1], d_nw[j,i] )
            U.low[J+1,I+1] = numpy.minimum( U.low[J+1,I+1], d_nw[j,i] )
            V.low[J+1,I] = numpy.minimum( V.low[J+1,I], d_nw[j,i] )
            V.low[J+1,I+1] = numpy.minimum( V.low[J+1,I+1], d_nw[j,i] )
            C.low[J,I] = numpy.minimum( C.low[J,I], d_nw[j,i] )
            C.low[J,I+1] = numpy.minimum( C.low[J,I+1], d_nw[j,i] )
            C.low[J+1,I] = numpy.minimum( C.low[J+1,I], d_nw[j,i] )
            C.low[J+1,I+1] = numpy.minimum( C.low[J+1,I+1], d_nw[j,i] )
        # Outer edges
        if matlab:
            new_ridge = numpy.minimum( r_ne, r_sw )
            V.low[J+2,I+1] = numpy.maximum( V.low[J+2,I+1], new_ridge[j,i] )
            U.low[J+1,I+2] = numpy.maximum( U.low[J+1,I+2], new_ridge[j,i] )
            U.low[J,I+2] = numpy.maximum( U.low[J,I+2], new_ridge[j,i] )
            V.low[J,I+1] = numpy.maximum( V.low[J,I+1], new_ridge[j,i] )
            V.low[J,I] = numpy.maximum( V.low[J,I], new_ridge[j,i] )
            U.low[J,I] = numpy.maximum( U.low[J,I], new_ridge[j,i] )
            new_ridge = 0.5 * ( r_ne + r_sw )
            V.ave[J+2,I+1] = numpy.maximum( V.ave[J+2,I+1], new_ridge[j,i] )
            U.ave[J+1,I+2] = numpy.maximum( U.ave[J+1,I+2], new_ridge[j,i] )
            U.ave[J,I+2] = numpy.maximum( U.ave[J,I+2], new_ridge[j,i] )
            V.ave[J,I+1] = numpy.maximum( V.ave[J,I+1], new_ridge[j,i] )
            V.ave[J,I] = numpy.maximum( V.ave[J,I], new_ridge[j,i] )
            U.ave[J,I] = numpy.maximum( U.ave[J,I], new_ridge[j,i] )
            new_ridge = numpy.maximum( r_ne, r_sw )
            V.hgh[J+2,I+1] = numpy.maximum( V.hgh[J+2,I+1], new_ridge[j,i] )
            U.hgh[J+1,I+2] = numpy.maximum( U.hgh[J+1,I+2], new_ridge[j,i] )
            U.hgh[J,I+2] = numpy.maximum( U.hgh[J,I+2], new_ridge[j,i] )
            V.hgh[J,I+1] = numpy.maximum( V.hgh[J,I+1], new_ridge[j,i] )
            V.hgh[J,I] = numpy.maximum( V.hgh[J,I], new_ridge[j,i] )
            U.hgh[J,I] = numpy.maximum( U.hgh[J,I], new_ridge[j,i] )
        else:
            new_ridge = numpy.minimum( r_ne, r_sw )
            V.low[J+2,I+1] = numpy.maximum( V.low[J+2,I+1], r_ne[j,i] )
            U.low[J+1,I+2] = numpy.maximum( U.low[J+1,I+2], r_ne[j,i] )
            U.low[J,I+2] = numpy.maximum( U.low[J,I+2], new_ridge[j,i] )
            V.low[J,I+1] = numpy.maximum( V.low[J,I+1], new_ridge[j,i] )
            V.low[J,I] = numpy.maximum( V.low[J,I], r_sw[j,i] )
            U.low[J,I] = numpy.maximum( U.low[J,I], r_sw[j,i] )
        if verbose: print("  NW: ", nwj.size, " inverted")

        # Apply NE
        j,i,J,I=nej,nei,2*nej,2*nei
        # Deepen interior walls and cells
        if matlab:
            U.low[J,I+1] = d_ne[j,i]
            U.low[J+1,I+1] = d_ne[j,i]
            V.low[J+1,I] = d_ne[j,i]
            V.low[J+1,I+1] = d_ne[j,i]
        else:
            U.low[J,I+1] = numpy.minimum( U.low[J,I+1], d_ne[j,i] )
            U.low[J+1,I+1] = numpy.minimum( U.low[J+1,I+1], d_ne[j,i] )
            V.low[J+1,I] = numpy.minimum( V.low[J+1,I], d_ne[j,i] )
            V.low[J+1,I+1] = numpy.minimum( V.low[J+1,I+1], d_ne[j,i] )
            C.low[J,I] = numpy.minimum( C.low[J,I], d_ne[j,i] )
            C.low[J,I+1] = numpy.minimum( C.low[J,I+1], d_ne[j,i] )
            C.low[J+1,I] = numpy.minimum( C.low[J+1,I], d_ne[j,i] )
            C.low[J+1,I+1] = numpy.minimum( C.low[J+1,I+1], d_ne[j,i] )
        # Outer edges
        if matlab:
            new_ridge = numpy.minimum( r_nw, r_se )
            V.low[J+2,I] = numpy.maximum( V.low[J+2,I], new_ridge[j,i] )
            U.low[J+1,I] = numpy.maximum( U.low[J+1,I], new_ridge[j,i] )
            U.low[J,I] = numpy.maximum( U.low[J,I], new_ridge[j,i] )
            V.low[J,I] = numpy.maximum( V.low[J,I], new_ridge[j,i] )
            V.low[J,I+1] = numpy.maximum( V.low[J,I+1], new_ridge[j,i] )
            U.low[J,I+2] = numpy.maximum( U.low[J,I+2], new_ridge[j,i] )
            new_ridge = 0.5 * ( r_nw + r_se )
            V.ave[J+2,I] = numpy.maximum( V.ave[J+2,I], new_ridge[j,i] )
            U.ave[J+1,I] = numpy.maximum( U.ave[J+1,I], new_ridge[j,i] )
            U.ave[J,I] = numpy.maximum( U.ave[J,I], new_ridge[j,i] )
            V.ave[J,I] = numpy.maximum( V.ave[J,I], new_ridge[j,i] )
            V.ave[J,I+1] = numpy.maximum( V.ave[J,I+1], new_ridge[j,i] )
            U.ave[J,I+2] = numpy.maximum( U.ave[J,I+2], new_ridge[j,i] )
            new_ridge = numpy.maximum( r_nw, r_se )
            V.hgh[J+2,I] = numpy.maximum( V.hgh[J+2,I], new_ridge[j,i] )
            U.hgh[J+1,I] = numpy.maximum( U.hgh[J+1,I], new_ridge[j,i] )
            U.hgh[J,I] = numpy.maximum( U.hgh[J,I], new_ridge[j,i] )
            V.hgh[J,I] = numpy.maximum( V.hgh[J,I], new_ridge[j,i] )
            V.hgh[J,I+1] = numpy.maximum( V.hgh[J,I+1], new_ridge[j,i] )
            U.hgh[J,I+2] = numpy.maximum( U.hgh[J,I+2], new_ridge[j,i] )
        else:
            new_ridge = numpy.minimum( r_nw, r_se )
            V.low[J+2,I] = numpy.maximum( V.low[J+2,I], r_nw[j,i] )
            U.low[J+1,I] = numpy.maximum( U.low[J+1,I], r_nw[j,i] )
            U.low[J,I] = numpy.maximum( U.low[J,I], new_ridge[j,i] )
            V.low[J,I] = numpy.maximum( V.low[J,I], new_ridge[j,i] )
            V.low[J,I+1] = numpy.maximum( V.low[J,I+1], r_se[j,i] )
            U.low[J,I+2] = numpy.maximum( U.low[J,I+2], r_se[j,i] )
        if verbose: print("  NE: ", nej.size, " inverted")
    def diagnose_EW_pathway(self, measure='effective'):
        """Returns deepest EW pathway"""
        wn_to_en, wn_to_es, ws_to_en, ws_to_es = self.diagnose_EW_pathways(measure=measure)
        wn = numpy.minimum( wn_to_en, wn_to_es)
        ws = numpy.minimum( ws_to_en, ws_to_es)
        return numpy.minimum( wn, ws)
    def diagnose_EW_pathways(self, measure='effective'):
        """Returns deepest EW pathway"""
        if measure == 'effective':
            self.u_effective.transpose()
            self.v_effective.transpose()
            self.u_effective,self.v_effective = self.v_effective,self.u_effective
        elif measure == 'simple':
            self.u_simple.transpose()
            self.v_simple.transpose()
            self.u_simple,self.v_simple = self.v_simple,self.u_simple
        else: raise Exception('Unknown "measure"')
        wn_to_en, wn_to_es, ws_to_en, ws_to_es = self.diagnose_NS_pathways(measure=measure)
        if measure == 'effective':
            self.u_effective.transpose()
            self.v_effective.transpose()
            self.u_effective,self.v_effective = self.v_effective,self.u_effective
        elif measure == 'simple':
            self.u_simple.transpose()
            self.v_simple.transpose()
            self.u_simple,self.v_simple = self.v_simple,self.u_simple
        else: raise Exception('Unknown "measure"')
        return wn_to_en.T, wn_to_es.T, ws_to_en.T, ws_to_es.T
    def diagnose_NS_pathway(self, measure='effective'):
        """Returns deepest NS pathway"""
        se_to_ne, se_to_nw, sw_to_ne, sw_to_nw = self.diagnose_NS_pathways(measure=measure)
        sw = numpy.minimum( sw_to_ne, sw_to_nw)
        se = numpy.minimum( se_to_ne, se_to_nw)
        return numpy.minimum( sw, se)
    def diagnose_NS_pathways(self, measure='effective'):
        """Returns NS deep pathways"""
        # Alias
        if measure == 'effective':
            C,U,V = self.c_effective.low,self.u_effective.low,self.v_effective.low
        elif measure == 'simple':
            C,U,V = self.c_simple.low,self.u_simple.low,self.v_simple.low
        else: raise Exception('Unknown "measure"')

        # Cell to immediate north-south exit
        ne_exit = V[2::2,1::2]
        nw_exit = V[2::2,::2]
        se_exit = V[:-1:2,1::2]
        sw_exit = V[:-1:2,::2]
        # Single gate cell to cell
        se_to_ne_1 = V[1::2,1::2]
        sw_to_nw_1 = V[1::2,::2]
        nw_to_ne_1 = U[1::2,1::2]
        ne_to_nw_1 = nw_to_ne_1
        sw_to_se_1 = U[::2,1::2]
        se_to_sw_1 = sw_to_se_1
        # Two gates cell to cell
        a = numpy.maximum( sw_to_se_1, se_to_ne_1 )
        b = numpy.maximum( sw_to_nw_1, nw_to_ne_1 )
        sw_to_ne = numpy.minimum( a, b )
        a = numpy.maximum( se_to_sw_1, sw_to_nw_1 )
        b = numpy.maximum( se_to_ne_1, ne_to_nw_1 )
        se_to_nw = numpy.minimum( a, b )
        # Both paths from south cells to north cells
        se_to_ne = numpy.maximum( se_to_nw, nw_to_ne_1 )
        se_to_ne = numpy.minimum( se_to_ne_1, se_to_ne )
        sw_to_nw = numpy.maximum( sw_to_ne, ne_to_nw_1 )
        sw_to_nw = numpy.minimum( sw_to_nw_1, sw_to_nw )
        # South cells to north exits (replaces previous definitions)
        se_to_ne = numpy.maximum( se_to_ne, ne_exit )
        se_to_nw = numpy.maximum( se_to_nw, nw_exit )
        sw_to_ne = numpy.maximum( sw_to_ne, ne_exit )
        sw_to_nw = numpy.maximum( sw_to_nw, nw_exit )
        # Entrance to exit (replaces previous definitions)
        se_to_ne = numpy.maximum( se_exit, se_to_ne )
        se_to_nw = numpy.maximum( se_exit, se_to_nw )
        sw_to_ne = numpy.maximum( sw_exit, sw_to_ne )
        sw_to_nw = numpy.maximum( sw_exit, sw_to_nw )

        return se_to_ne, se_to_nw, sw_to_ne, sw_to_nw
    def limit_NS_EW_connections(self, ns_deepest_connection, ew_deepest_connection, verbose=False):
        """Modify outer edges to satisfy NS and EW deepest connections"""
        # Alias
        U,V = self.u_effective.low,self.v_effective.low
        n = numpy.minimum( V[2::2,::2], V[2::2,1::2] )
        s = numpy.minimum( V[:-1:2,::2], V[:-1:2,1::2] )
        e = numpy.minimum( U[::2,2::2], U[1::2,2::2] )
        w = numpy.minimum( U[::2,:-1:2], U[1::2,:-1:2] )

        if verbose: print("limit_NS_EW_connections")
        needed = ns_deepest_connection > numpy.maximum( n, s )
        j,i = numpy.nonzero( needed & ( s>=n ) ); J,I=2*j,2*i
        if verbose: print("  S: ", j.size, " raised")
        V[J,I] = numpy.maximum( V[J,I], ns_deepest_connection[j,i] )
        V[J,I+1] = numpy.maximum( V[J,I+1], ns_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( s<=n ) ); J,I=2*j,2*i
        if verbose: print("  N: ", j.size, " raised")
        V[J+2,I] = numpy.maximum( V[J+2,I], ns_deepest_connection[j,i] )
        V[J+2,I+1] = numpy.maximum( V[J+2,I+1], ns_deepest_connection[j,i] )

        needed = ew_deepest_connection > numpy.maximum( e, w )
        j,i = numpy.nonzero( needed & ( w>=e ) ); J,I=2*j,2*i
        if verbose: print("  W: ", j.size, " raised")
        U[J,I] = numpy.maximum( U[J,I], ew_deepest_connection[j,i] )
        U[J+1,I] = numpy.maximum( U[J+1,I], ew_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( w<=e ) ); J,I=2*j,2*i
        if verbose: print("  E: ", j.size, " raised")
        U[J,I+2] = numpy.maximum( U[J,I+2], ew_deepest_connection[j,i] )
        U[J+1,I+2] = numpy.maximum( U[J+1,I+2], ew_deepest_connection[j,i] )
    def diagnose_corner_pathways(self, measure='effective'):
        """Returns deepest corner pathways"""
        sw = self.diagnose_SW_pathway(measure=measure)
        # Alias
        if measure == 'effective':
            C,U,V = self.c_effective,self.u_effective,self.v_effective
        elif measure == 'simple':
            C,U,V = self.c_simple,self.u_simple,self.v_simple
        else: raise Exception('Unknown "measure"')

        # Flip in j direction so j=S, i=E
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        nw = self.diagnose_SW_pathway(measure=measure)
        nw = numpy.flip(nw, axis=0)
        # Flip in i direction so j=S, i=W
        C.flip(axis=1)
        U.flip(axis=1)
        V.flip(axis=1)
        ne = self.diagnose_SW_pathway(measure=measure)
        ne = numpy.flip(numpy.flip(ne, axis=0), axis=1)
        # Flip in j direction so j=N, i=W
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        se = self.diagnose_SW_pathway(measure=measure)
        se = numpy.flip(se, axis=1)
        # Flip in i direction so j=N, i=E
        C.flip(axis=1)
        U.flip(axis=1)
        V.flip(axis=1)
        return sw, se, ne, nw
    def diagnose_SW_pathway(self, measure='effective'):
        """Returns deepest SW pathway"""
        sw_to_sw, sw_to_nw, se_to_sw, se_to_nw = self.diagnose_SW_pathways(measure=measure)
        sw = numpy.minimum( sw_to_sw, sw_to_nw)
        se = numpy.minimum( se_to_sw, se_to_nw)
        return numpy.minimum( sw, se)
    def diagnose_SW_pathways(self, measure='effective'):
        """Returns SW deep pathways"""
        # Alias
        if measure == 'effective':
            C,U,V = self.c_effective.low,self.u_effective.low,self.v_effective.low
        elif measure == 'simple':
            C,U,V = self.c_simple.low,self.u_simple.low,self.v_simple.low
        else: raise Exception('Unknown "measure"')

        # Cell to immediate south/west exit
        w_n_exit = U[1::2,:-1:2]
        w_s_exit = U[::2,:-1:2]
        s_e_exit = V[:-1:2,1::2]
        s_w_exit = V[:-1:2,::2]

        # Single gate ell to cell
        se_to_ne_1 = V[1::2,1::2]
        sw_to_nw_1 = V[1::2,::2]
        nw_to_sw_1 = sw_to_nw_1
        ne_to_nw_1 = U[1::2,1::2]
        nw_to_ne_1 = ne_to_nw_1
        se_to_sw_1 = U[::2,1::2]
        sw_to_se_1 = se_to_sw_1

        se_to_nw_via_ne = numpy.maximum( se_to_ne_1, ne_to_nw_1 )

        sw_to_nw = numpy.maximum( sw_to_se_1, se_to_nw_via_ne )
        sw_to_nw = numpy.minimum( sw_to_nw_1, sw_to_nw )

        se_to_nw_via_sw = numpy.maximum( se_to_sw_1, sw_to_nw_1 )
        se_to_nw = numpy.minimum( se_to_nw_via_sw, se_to_nw_via_ne )

        se_to_sw = numpy.maximum( se_to_nw_via_ne, nw_to_sw_1 )
        se_to_sw = numpy.minimum( se_to_sw, se_to_sw_1 )

        # Entrance to exit
        sw_to_sw = numpy.maximum( s_w_exit, w_s_exit )
        sw_to_nw = numpy.maximum( sw_to_nw, w_n_exit )
        sw_to_nw = numpy.maximum( sw_to_nw, s_w_exit )
        se_to_sw = numpy.maximum( se_to_sw, w_s_exit )
        se_to_sw = numpy.maximum( se_to_sw, s_e_exit )
        se_to_nw = numpy.maximum( se_to_nw, w_n_exit )
        se_to_nw = numpy.maximum( se_to_nw, s_e_exit )

        return sw_to_sw, sw_to_nw, se_to_sw, se_to_nw
    def limit_corner_connections(self, sw_deepest_connection, se_deepest_connection, ne_deepest_connection, nw_deepest_connection, verbose=False):
        """Modify outer edges to satisfy deepest corner connections"""
        # Alias
        U, V = self.u_effective.low, self.v_effective.low
        n = numpy.minimum( V[2::2,::2], V[2::2,1::2] )
        s = numpy.minimum( V[:-1:2,::2], V[:-1:2,1::2] )
        e = numpy.minimum( U[::2,2::2], U[1::2,2::2] )
        w = numpy.minimum( U[::2,:-1:2], U[1::2,:-1:2] )

        if verbose: print('limit_corner_connections')
        needed = sw_deepest_connection > numpy.maximum( s, w )
        j,i = numpy.nonzero( needed & ( s>=w ) ); J,I=2*j,2*i
        V[J,I] = numpy.maximum( V[J,I], sw_deepest_connection[j,i] )
        V[J,I+1] = numpy.maximum( V[J,I+1], sw_deepest_connection[j,i] )
        if verbose: print("  SW corner S: ", j.size, " raised")
        j,i = numpy.nonzero( needed & ( s<=w ) ); J,I=2*j,2*i
        U[J,I] = numpy.maximum( U[J,I], sw_deepest_connection[j,i] )
        U[J+1,I] = numpy.maximum( U[J+1,I], sw_deepest_connection[j,i] )
        if verbose: print("  SW corner W: ", j.size, " raised")

        needed = se_deepest_connection > numpy.maximum( s, e )
        j,i = numpy.nonzero( needed & ( s>=e ) ); J,I=2*j,2*i
        if verbose: print("  SE corner S: ", j.size, " raised")
        V[J,I] = numpy.maximum( V[J,I], se_deepest_connection[j,i] )
        V[J,I+1] = numpy.maximum( V[J,I+1], se_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( s<=e ) ); J,I=2*j,2*i
        if verbose: print("  SE corner E: ", j.size, " raised")
        U[J,I+2] = numpy.maximum( U[J,I+2], se_deepest_connection[j,i] )
        U[J+1,I+2] = numpy.maximum( U[J+1,I+2], se_deepest_connection[j,i] )

        needed = ne_deepest_connection > numpy.maximum( n, e )
        j,i = numpy.nonzero( needed & ( n>=e ) ); J,I=2*j,2*i
        if verbose: print("  NE corner N: ", j.size, " raised")
        V[J+2,I] = numpy.maximum( V[J+2,I], ne_deepest_connection[j,i] )
        V[J+2,I+1] = numpy.maximum( V[J+2,I+1], ne_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( n<=e ) ); J,I=2*j,2*i
        if verbose: print("  NE corner E: ", j.size, " raised")
        U[J,I+2] = numpy.maximum( U[J,I+2], ne_deepest_connection[j,i] )
        U[J+1,I+2] = numpy.maximum( U[J+1,I+2], ne_deepest_connection[j,i] )

        needed = nw_deepest_connection > numpy.maximum( n, w )
        j,i = numpy.nonzero( needed & ( n>=w ) ); J,I=2*j,2*i
        if verbose: print("  NW corner N: ", j.size, " raised")
        V[J+2,I] = numpy.maximum( V[J+2,I], nw_deepest_connection[j,i] )
        V[J+2,I+1] = numpy.maximum( V[J+2,I+1], nw_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( n<=w ) ); J,I=2*j,2*i
        if verbose: print("  NW corner W: ", j.size, " raised")
        U[J,I] = numpy.maximum( U[J,I], nw_deepest_connection[j,i] )
        U[J+1,I] = numpy.maximum( U[J+1,I], nw_deepest_connection[j,i] )

    def lift_ave_max(self):
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        C.ave, U.ave, V.ave = numpy.maximum(C.ave, C.low), numpy.maximum(U.ave, U.low), numpy.maximum(V.ave, V.low)
        C.hgh, U.hgh, V.hgh = numpy.maximum(C.hgh, C.ave), numpy.maximum(U.hgh, U.ave), numpy.maximum(V.hgh, V.ave)

    def coarsen(self, do_thinwalls=True, do_effective=True):
        M = ThinWalls(lon=self.lon[::2,::2],lat=self.lat[::2,::2],rfl=self.rfl-1)
        M.c_simple.ave = self.c_simple.mean4()
        M.c_simple.low = self.c_simple.min4()
        M.c_simple.hgh = self.c_simple.max4()
        if do_thinwalls:
            M.u_simple.ave = self.u_simple.mean2u()
            M.u_simple.low = self.u_simple.min2u()
            M.u_simple.hgh = self.u_simple.max2u()
            M.v_simple.ave = self.v_simple.mean2v()
            M.v_simple.low = self.v_simple.min2v()
            M.v_simple.hgh = self.v_simple.max2v()
            if do_effective:
                M.c_effective.ave = self.c_effective.mean4()
                M.c_effective.low = self.c_effective.min4()
                M.c_effective.hgh = self.c_effective.max4()
                M.u_effective.ave = self.u_effective.mean2u()
                M.u_effective.low = self.u_effective.min2u()
                M.u_effective.hgh = self.u_effective.max2u()
                M.v_effective.ave = self.v_effective.mean2v()
                M.v_effective.low = self.v_effective.min2v()
                M.v_effective.hgh = self.v_effective.max2v()
        return M

    def boundHbyUV(self):
        """Bound center values to be lower than edge values"""
        # for coarsened grid
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        He = numpy.minimum( numpy.minimum(U.low[:,:-1], U.low[:,1:]), numpy.minimum(V.low[:-1,:], V.low[1:,:]) )
        C.low = numpy.minimum(C.low, He)

        U.ave = numpy.maximum(U.low, U.ave)
        V.ave = numpy.maximum(V.low, V.ave)
        He = numpy.minimum( numpy.minimum(U.ave[:,:-1], U.ave[:,1:]), numpy.minimum(V.ave[:-1,:], V.ave[1:,:]) )
        C.ave = numpy.minimum(C.ave, He)

        U.hgh = numpy.maximum(U.ave, U.hgh)
        V.hgh = numpy.maximum(V.ave, V.hgh)
        # # why not bound C.hgh???
        # He = numpy.minimum( numpy.minimum(U.hgh[:,:-1], U.hgh[:,1:]), numpy.minimum(V.hgh[:-1,:], V.hgh[1:,:]) )
        # C.hgh = numpy.minimum(C.hgh, He)

    def regenUV(self):
        pass

    def fillPotHoles(self):
        """Bound center values to be higher than edge values"""
        # for coarsened grid
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        He = numpy.minimum( numpy.minimum(U.low[:,:-1], U.low[:,1:]), numpy.minimum(V.low[:-1,:], V.low[1:,:]) )
        C.low = numpy.maximum(C.low, He)

        He = numpy.minimum( numpy.minimum(U.ave[:,:-1], U.ave[:,1:]), numpy.minimum(V.ave[:-1,:], V.ave[1:,:]) )
        C.ave = numpy.maximum(C.ave, He)

    def plot(self, axis, thickness=0.2, metric='mean', measure='simple', *args, **kwargs):
        """Plots ThinWalls data."""
        def copy_coord(xy):
            XY = numpy.zeros( (2*self.nj+2,2*self.ni+2) )
            dr = xy[1:,1:] - xy[:-1,:-1]
            dl = xy[:-1,1:] - xy[1:,:-1]

            XY[::2,::2] = xy
            # Reference to the northeast corner of the cell located to the southwest
            XY[2::2,2::2] = XY[2::2,2::2] - dr*thickness/2
            # Southmost row
            XY[0,::2] = XY[0,::2] - numpy.r_[dr[0,:], dr[0,-1]]*thickness/2
            # Westmost column (excluding the southwestmost point)
            XY[2::2,0] = XY[2::2,0] - numpy.r_[dr[1:,0] ,dr[-1,0]]*thickness/2

            XY[1::2,::2] = xy
            # Reference to the southeast corner of the cell located to the northwest
            XY[1:-1:2,2::2] = XY[1:-1:2,2::2] - dl*thickness/2
            # Westmost column
            XY[1::2,0] = XY[1::2,0] - numpy.r_[dl[0,0],  dl[:,0]]*thickness/2
            # Northmost row (excluding the northwestmost point)
            XY[-1,2::2] = XY[-1,2::2] - numpy.r_[dl[-1,1:], dl[-1,-1]]*thickness/2

            XY[::2,1::2] = xy
            # Reference to the northwest corner of the cell located to the southeast
            XY[2::2,1:-1:2] = XY[2::2,1:-1:2] + dl*thickness/2
            # Eastmost column
            XY[::2,-1] = XY[::2,-1] + numpy.r_[dl[:,-1], dl[-1,-1]]*thickness/2
            # Southmost row (excluding the southeastmost point)
            XY[0,1:-1:2] = XY[0,1:-1:2] + numpy.r_[dl[0,0], dl[0,:-1]]*thickness/2

            XY[1::2,1::2] = xy
            # Reference to the southwest corner of the cell located to the northeast
            XY[1:-1:2,1:-1:2] = XY[1:-1:2,1:-1:2] + dr*thickness/2
            # Northmost row
            XY[-1,1::2] = XY[-1,1::2] + numpy.r_[dr[-1,0], dr[-1,:]]*thickness/2
            # Eastmost column (excluding the northeastmost point)
            XY[1:-1:2,-1] = XY[1:-1:2,-1] + numpy.r_[dr[0,-1], dr[:-1,-1]]*thickness/2

            return XY
        lon = copy_coord(self.lon)
        lat = copy_coord(self.lat)
        def pcol_elev(c,u,v):
            tmp = numpy.ma.zeros( (2*self.nj+1,2*self.ni+1) )
            tmp[::2,::2] = numpy.ma.masked # Mask corner values
            tmp[1::2,1::2] = c
            tmp[1::2,::2] = u
            tmp[::2,1::2] = v
            return axis.pcolormesh(lon, lat, tmp, *args, **kwargs)
        if measure=='simple':
            c,u,v = self.c_simple, self.u_simple, self.v_simple
        elif measure=='effective':
            c,u,v = self.c_effective, self.u_effective, self.v_effective
        else: raise Exception('Unknown "measure"')
        if metric=='mean': return pcol_elev( c.ave, u.ave, v.ave )
        elif metric=='min': return pcol_elev( c.low, u.low, v.low )
        elif metric=='max': return pcol_elev( c.hgh, u.hgh, v.hgh )
        else: raise Exception('Unknown "metric"')
    def plot_grid(self, axis, *args, **kwargs):
        """Plots ThinWalls mesh."""
        super().plot(axis, *args, **kwargs)
