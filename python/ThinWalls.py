from GMesh import GMesh
import numpy

class Stats:
    """Container for statistics fields

    shape - shape of these arrays
    low   - minimum value
    hgh   - maximum value
    ave   - mean value
    """
    def __init__(self, shape, mean=None, min=None, max=None):
        self.shape = shape
        self.low = numpy.zeros(shape)
        self.hgh = numpy.zeros(shape)
        self.ave = numpy.zeros(shape)
        if (mean is not None) and (min is not None) and (max is not None):
            self.set(min, max, mean)
        else:
            if mean is not None: self.set_equal(mean)
            if min is not None: self.set_equal(min)
            if max is not None: self.set_equal(max)
    def __repr__(self):
        return '<Stats shape:(%i,%i)>'%(self.shape[0], self.shape[1])
    def __copy__(self):
        return Stats(self.shape, mean=self.ave, min=self.low, max=self.hgh)
    def __getitem__(self, key):
        key = Stats._convert_key_to_slice(key)

        low = self.low[key]
        hgh = self.hgh[key]
        ave = self.ave[key]
        shape = ave.shape
        return Stats(shape, mean=ave, min=low, max=hgh)
    def __setitem__(self, key, value):
        key = Stats._convert_key_to_slice(key)

        self.low[key] = value.low[:]
        self.hgh[key] = value.hgh[:]
        self.ave[key] = value.ave[:]
    @staticmethod
    def _convert_key_to_slice(key):
        """Convert a two-element array key to a tuple of two slices
        This makes sure the sliced Stats is always 2D.
        """
        def single_key_to_slice(key):
            if key==-1:
                key_slice = slice(key,None)
            else:
                key_slice = slice(key,key+1)
            return key_slice
        if isinstance(key, tuple):
            key = list(key)
            if not isinstance(key[0],slice): key[0] = single_key_to_slice(key[0])
            if not isinstance(key[1],slice): key[1] = single_key_to_slice(key[1])
            return tuple(key)
        else:
            return single_key_to_slice(key)
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
        self.low = self.low.T
        self.ave = self.ave.T
        self.hgh = self.hgh.T
        self.shape = self.low.shape

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
    def set_edge_to_step(self):
        """Set elevation of cell edges to step topography."""
        tmp = numpy.zeros(self.shapeu)
        tmp[:,1:-1] = numpy.maximum( self.c_simple.ave[:,:-1], self.c_simple.ave[:,1:] )
        tmp[:,0] = self.c_simple.ave[:,0]
        tmp[:,-1] = self.c_simple.ave[:,-1]
        self.u_simple.set_equal( tmp )
        tmp = numpy.zeros(self.shapev)
        tmp[1:-1,:] = numpy.maximum( self.c_simple.ave[:-1,:], self.c_simple.ave[1:,:] )
        tmp[0,:] = self.c_simple.ave[0,:]
        tmp[-1,:] = self.c_simple.ave[-1,:]
        self.v_simple.set_equal( tmp )
    def set_center_from_corner(self):
        """Set elevation of cell centers from corners."""
        self.c_simple.ave = 0.25 * ( (self.height[:-1,:-1] + self.height[1:,1:])
                                    +(self.height[1:,:-1] + self.height[:-1,1:]) )
        self.c_simple.hgh = numpy.maximum( numpy.maximum( self.height[:-1,:-1], self.height[1:,1:]),
                                           numpy.maximum( self.height[1:,:-1], self.height[:-1,1:]) )
        self.c_simple.low = numpy.minimum( numpy.minimum( self.height[:-1,:-1], self.height[1:,1:]),
                                           numpy.minimum( self.height[1:,:-1], self.height[:-1,1:]) )
    def set_edge_from_corner(self):
        """Set elevation of cell edges from corners."""
        self.u_simple.ave = 0.5 * ( self.height[:-1,:] + self.height[1:,:] )
        self.u_simple.hgh = numpy.maximum( self.height[:-1,:], self.height[1:,:] )
        self.u_simple.low = numpy.minimum( self.height[:-1,:], self.height[1:,:] )
        self.v_simple.ave = 0.5 * ( self.height[:,:-1] + self.height[:,1:] )
        self.v_simple.hgh = numpy.maximum( self.height[:,:-1], self.height[:,1:] )
        self.v_simple.low = numpy.minimum( self.height[:,:-1], self.height[:,1:] )
    def push_corners(self, update_interior_mean_max=True, verbose=False):
        """Folds out tallest corners. Acts only on "effective" values.

        A convex corner within a coarse grid cell can be made into a
        concave corner without changing connectivity across the major
        parts of the cell. The cross-corner connection for the minor
        part of the cell is eliminated."""

        if verbose: print("Begin push_corners")
        if verbose: print("  SW: ", end="")
        self.push_corners_sw(update_interior_mean_max=update_interior_mean_max, verbose=verbose) # Push SW
        # Alias
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        # Flip in j direction
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        if verbose: print("  NW: ", end="")
        self.push_corners_sw(update_interior_mean_max=update_interior_mean_max, verbose=verbose) # Push NW
        # Flip in i direction
        C.flip(axis=1)
        U.flip(axis=1)
        V.flip(axis=1)
        if verbose: print("  NE: ", end="")
        self.push_corners_sw(update_interior_mean_max=update_interior_mean_max, verbose=verbose) # Push NE
        # Flip in j direction
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        if verbose: print("  SE: ", end="")
        self.push_corners_sw(update_interior_mean_max=update_interior_mean_max, verbose=verbose) # Push SE
        # Flip in i direction
        C.flip(axis=1)
        U.flip(axis=1)
        V.flip(axis=1)
    def push_corners_sw(self, update_interior_mean_max=True, verbose=False):
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
    def lower_tallest_buttress(self, verbose=False):
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
    def fold_out_central_ridges(self, verbose=False):
        """Folded out interior ridges to the sides of the coarse cell"""
        if verbose: print("Begin fold_out_central_ridges")
        if verbose: print("  S: ", end="")
        self.fold_out_central_ridge_s(verbose=verbose)
        # Alias
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        # Flip in j direction so j=S, i=E
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        if verbose: print("  N: ", end="")
        self.fold_out_central_ridge_s(verbose=verbose)
        # Transpose so j=E, i=S
        C.transpose()
        U.transpose()
        V.transpose()
        self.u_effective,self.v_effective = self.v_effective,self.u_effective
        C, U, V = self.c_effective, self.u_effective, self.v_effective
        if verbose: print("  W: ", end="")
        self.fold_out_central_ridge_s(verbose=verbose)
        # Flip in j direction so j=W, i=S
        C.flip(axis=0)
        U.flip(axis=0)
        V.flip(axis=0)
        if verbose: print("  E: ", end="")
        self.fold_out_central_ridge_s(verbose=verbose)
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
    def fold_out_central_ridge_s(self, verbose=False):
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
              ( ( ew_ridge_low>ns_ridge_low_min) & (ew_ridge_low>=ns_ridge_low_max ) ) # E-W ridge is the taller ridge
              & (
                  ( U.low[::2,1::2] > U.low[1::2,1::2] ) # Southern buttress is taller than north
                  | (
                      ( U.low[::2,1::2] >= U.low[1::2,1::2] ) # Southern buttress is equal to the north
                      & (
                          ( C.low[::2,::2]+C.low[::2,1::2] > C.low[1::2,::2]+C.low[1::2,1::2] ) | # Southern cells are higher than north on average
                          ( V.low[:-1:2,::2]+V.low[:-1:2,1::2] > V.low[2::2,::2]+V.low[2::2,1::2] ) # Southern edges are higher than north on average
                ) ) ) )
        J,I = 2*j,2*i
        # Outer edges of southern half
        U.low[J,I] = numpy.maximum( U.low[J,I], ew_ridge_low[j,i] )
        V.low[J,I] = numpy.maximum( V.low[J,I], ew_ridge_low[j,i] )
        V.low[J,I+1] = numpy.maximum( V.low[J,I+1], ew_ridge_low[j,i] )
        U.low[J,I+2] = numpy.maximum( U.low[J,I+2], ew_ridge_low[j,i] )
        # Replace E-W ridge
        V.low[J+1,I] = ns_ridge_low_min[j,i]
        V.low[J+1,I+1] = ns_ridge_low_min[j,i]
        # Southern cells
        C.low[J,I] = ns_ridge_low_min[j,i]
        C.low[J,I+1] = ns_ridge_low_min[j,i]
        U.low[J,I+1] = ns_ridge_low_min[j,i]
        if verbose: print(j.size, " folded")
    def invert_exterior_corners(self, verbose=False):
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
        U.low[J,I+1] = numpy.minimum( U.low[J,I+1], d_sw[j,i] )
        U.low[J+1,I+1] = numpy.minimum( U.low[J+1,I+1], d_sw[j,i] )
        V.low[J+1,I] = numpy.minimum( V.low[J+1,I], d_sw[j,i] )
        V.low[J+1,I+1] = numpy.minimum( V.low[J+1,I+1], d_sw[j,i] )
        C.low[J,I] = numpy.minimum( C.low[J,I], d_sw[j,i] )
        C.low[J,I+1] = numpy.minimum( C.low[J,I+1], d_sw[j,i] )
        C.low[J+1,I] = numpy.minimum( C.low[J+1,I], d_sw[j,i] )
        C.low[J+1,I+1] = numpy.minimum( C.low[J+1,I+1], d_sw[j,i] )
        # Outer edges
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
        U.low[J,I+1] = numpy.minimum( U.low[J,I+1], d_se[j,i] )
        U.low[J+1,I+1] = numpy.minimum( U.low[J+1,I+1], d_se[j,i] )
        V.low[J+1,I] = numpy.minimum( V.low[J+1,I], d_se[j,i] )
        V.low[J+1,I+1] = numpy.minimum( V.low[J+1,I+1], d_se[j,i] )
        C.low[J,I] = numpy.minimum( C.low[J,I], d_se[j,i] )
        C.low[J,I+1] = numpy.minimum( C.low[J,I+1], d_se[j,i] )
        C.low[J+1,I] = numpy.minimum( C.low[J+1,I], d_se[j,i] )
        C.low[J+1,I+1] = numpy.minimum( C.low[J+1,I+1], d_se[j,i] )
        # Outer edges
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
        U.low[J,I+1] = numpy.minimum( U.low[J,I+1], d_nw[j,i] )
        U.low[J+1,I+1] = numpy.minimum( U.low[J+1,I+1], d_nw[j,i] )
        V.low[J+1,I] = numpy.minimum( V.low[J+1,I], d_nw[j,i] )
        V.low[J+1,I+1] = numpy.minimum( V.low[J+1,I+1], d_nw[j,i] )
        C.low[J,I] = numpy.minimum( C.low[J,I], d_nw[j,i] )
        C.low[J,I+1] = numpy.minimum( C.low[J,I+1], d_nw[j,i] )
        C.low[J+1,I] = numpy.minimum( C.low[J+1,I], d_nw[j,i] )
        C.low[J+1,I+1] = numpy.minimum( C.low[J+1,I+1], d_nw[j,i] )
        # Outer edges
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
        U.low[J,I+1] = numpy.minimum( U.low[J,I+1], d_ne[j,i] )
        U.low[J+1,I+1] = numpy.minimum( U.low[J+1,I+1], d_ne[j,i] )
        V.low[J+1,I] = numpy.minimum( V.low[J+1,I], d_ne[j,i] )
        V.low[J+1,I+1] = numpy.minimum( V.low[J+1,I+1], d_ne[j,i] )
        C.low[J,I] = numpy.minimum( C.low[J,I], d_ne[j,i] )
        C.low[J,I+1] = numpy.minimum( C.low[J,I+1], d_ne[j,i] )
        C.low[J+1,I] = numpy.minimum( C.low[J+1,I], d_ne[j,i] )
        C.low[J+1,I+1] = numpy.minimum( C.low[J+1,I+1], d_ne[j,i] )
        # Outer edges
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
    def limit_NS_EW_connections(self, ns_deepest_connection, ew_deepest_connection):
        """Modify outer edges to satisfy NS and EW deepest connections"""
        # Alias
        U,V = self.u_effective.low,self.v_effective.low
        n = numpy.minimum( V[2::2,::2], V[2::2,1::2] )
        s = numpy.minimum( V[:-1:2,::2], V[:-1:2,1::2] )
        e = numpy.minimum( U[::2,2::2], U[1::2,2::2] )
        w = numpy.minimum( U[::2,:-1:2], U[1::2,:-1:2] )

        needed = ns_deepest_connection > numpy.maximum( n, s )
        j,i = numpy.nonzero( needed & ( s>=n ) ); J,I=2*j,2*i
        V[J,I] = numpy.maximum( V[J,I], ns_deepest_connection[j,i] )
        V[J,I+1] = numpy.maximum( V[J,I+1], ns_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( s<=n ) ); J,I=2*j,2*i
        V[J+2,I] = numpy.maximum( V[J+2,I], ns_deepest_connection[j,i] )
        V[J+2,I+1] = numpy.maximum( V[J+2,I+1], ns_deepest_connection[j,i] )

        needed = ew_deepest_connection > numpy.maximum( e, w )
        j,i = numpy.nonzero( needed & ( w>=e ) ); J,I=2*j,2*i
        U[J,I] = numpy.maximum( U[J,I], ew_deepest_connection[j,i] )
        U[J+1,I] = numpy.maximum( U[J+1,I], ew_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( w<=e ) ); J,I=2*j,2*i
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
    def limit_corner_connections(self, sw_deepest_connection, se_deepest_connection, ne_deepest_connection, nw_deepest_connection):
        """Modify outer edges to satisfy deepest corner connections"""
        # Alias
        U, V = self.u_effective.low, self.v_effective.low
        n = numpy.minimum( V[2::2,::2], V[2::2,1::2] )
        s = numpy.minimum( V[:-1:2,::2], V[:-1:2,1::2] )
        e = numpy.minimum( U[::2,2::2], U[1::2,2::2] )
        w = numpy.minimum( U[::2,:-1:2], U[1::2,:-1:2] )

        needed = sw_deepest_connection > numpy.maximum( s, w )
        j,i = numpy.nonzero( needed & ( s>=w ) ); J,I=2*j,2*i
        V[J,I] = numpy.maximum( V[J,I], sw_deepest_connection[j,i] )
        V[J,I+1] = numpy.maximum( V[J,I+1], sw_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( s<=w ) ); J,I=2*j,2*i
        U[J,I] = numpy.maximum( U[J,I], sw_deepest_connection[j,i] )
        U[J+1,I] = numpy.maximum( U[J+1,I], sw_deepest_connection[j,i] )

        needed = se_deepest_connection > numpy.maximum( s, e )
        j,i = numpy.nonzero( needed & ( s>=e ) ); J,I=2*j,2*i
        V[J,I] = numpy.maximum( V[J,I], se_deepest_connection[j,i] )
        V[J,I+1] = numpy.maximum( V[J,I+1], se_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( s<=e ) ); J,I=2*j,2*i
        U[J,I+2] = numpy.maximum( U[J,I+2], se_deepest_connection[j,i] )
        U[J+1,I+2] = numpy.maximum( U[J+1,I+2], se_deepest_connection[j,i] )

        needed = ne_deepest_connection > numpy.maximum( n, e )
        j,i = numpy.nonzero( needed & ( n>=e ) ); J,I=2*j,2*i
        V[J+2,I] = numpy.maximum( V[J+2,I], ne_deepest_connection[j,i] )
        V[J+2,I+1] = numpy.maximum( V[J+2,I+1], ne_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( n<=e ) ); J,I=2*j,2*i
        U[J,I+2] = numpy.maximum( U[J,I+2], ne_deepest_connection[j,i] )
        U[J+1,I+2] = numpy.maximum( U[J+1,I+2], ne_deepest_connection[j,i] )

        needed = nw_deepest_connection > numpy.maximum( n, w )
        j,i = numpy.nonzero( needed & ( n>=w ) ); J,I=2*j,2*i
        V[J+2,I] = numpy.maximum( V[J+2,I], nw_deepest_connection[j,i] )
        V[J+2,I+1] = numpy.maximum( V[J+2,I+1], nw_deepest_connection[j,i] )
        j,i = numpy.nonzero( needed & ( n<=w ) ); J,I=2*j,2*i
        U[J,I] = numpy.maximum( U[J,I], nw_deepest_connection[j,i] )
        U[J+1,I] = numpy.maximum( U[J+1,I], nw_deepest_connection[j,i] )

    def coarsen(self):
        M = ThinWalls(lon=self.lon[::2,::2],lat=self.lat[::2,::2],rfl=self.rfl-1)
        M.c_simple.ave = self.c_simple.mean4()
        M.c_simple.low = self.c_simple.min4()
        M.c_simple.hgh = self.c_simple.max4()
        M.u_simple.ave =self.u_simple.mean2u()
        M.u_simple.low =self.u_simple.min2u()
        M.u_simple.hgh =self.u_simple.max2u()
        M.v_simple.ave = self.v_simple.mean2v()
        M.v_simple.low = self.v_simple.min2v()
        M.v_simple.hgh = self.v_simple.max2v()
        M.c_effective.ave = self.c_effective.mean4()
        M.c_effective.low = self.c_effective.min4()
        M.c_effective.hgh = self.c_effective.max4()
        M.u_effective.ave =self.u_effective.mean2u()
        M.u_effective.low =self.u_effective.min2u()
        M.u_effective.hgh =self.u_effective.max2u()
        M.v_effective.ave = self.v_effective.mean2v()
        M.v_effective.low = self.v_effective.min2v()
        M.v_effective.hgh = self.v_effective.max2v()
        return M
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
