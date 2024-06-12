#!/usr/bin/env python

import numpy as np
import time

def is_coord_uniform(coord, tol=1.e-5):
    """Returns True if the coordinate "coord" is uniform along the first axis, and False otherwise

    tol is the allowed fractional variation in spacing, i.e. ( variation in delta ) / delta < tol"""
    eps = np.finfo( coord.dtype ).eps # Precision of datatype
    # abscoord = np.abs( coord ) # Magnitude of coordinate values
    # abscoord = np.maximum( abscoord[1:], abscoord[:-1] ) # Largest magnitude of coordinate used at centers
    # roundoff = eps * abscoord # This is the roundoff error in calculating "coord[1:] - coord[:-1]"
    delta = np.abs( coord[1:] - coord[:-1] ) # Spacing along first axis
    roundoff = tol * delta[0] # Assuming delta is approximately uniform, use first value to estimate allowed variance
    derror = np.abs( delta - delta.flatten()[0] ) # delta should be uniform so delta - delta[0] should be zero
    return np.all( derror <= roundoff )

def is_mesh_uniform(lon,lat):
    """Returns True if the input grid (lon,lat) is uniform and False otherwise"""
    assert len(lon.shape) == len(lat.shape), "Arguments lon and lat must have the same rank"
    if len(lon.shape)==2: # 2D array
        assert lon.shape == lat.shape, "Arguments lon and lat must have the same shape"
    assert len(lon.shape)<3 and len(lat.shape)<3, "Arguments must be either both be 1D or both be 2D arralat"
    return is_coord_uniform(lat) and is_coord_uniform(lon.T)

def pfactor(n):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 0] # 0 causes error
    for p in primes:
        assert p > 0, "Ran out of primes - use a more robust method ..."
        if n % p == 0:
            if n == p:
                return [ p ]
            else:
                x = pfactor( n // p )
                x.append( p )
                return x
        if p * p > n:
            return [ n ]
    return [ n ]

class GMesh:
    """Describes 2D meshes for ESMs.

    Meshes have shape=(nj,ni) cells with (nj+1,ni+1) vertices with coordinates (lon,lat).

    When constructing, either provide 1d or 2d coordinates (lon,lat), or assume a
    uniform spherical grid with 'shape' cells covering the whole sphere with
    longitudes starting at lon0.

    Attributes:

    shape - (nj,ni)
    ni    - number of cells in i-direction (last)
    nj    - number of cells in j-direction (first)
    lon   - longitude of mesh (cell corners), shape (nj+1,ni=1)
    lat   - latitude of mesh (cell corners), shape (nj+1,ni=1)
    area  - area of cells, shape (nj,ni)
    """

    def __init__(self, shape=None, lon=None, lat=None, area=None, lon0=-180., from_cell_center=False, is_geo_coord=True, rfl=0):
        """Constructor for Mesh:
        shape - shape of cell array, (nj,ni)
        ni    - number of cells in i-direction (last index)
        nj    - number of cells in j-direction (first index)
        lon   - longitude of mesh (cell corners) (1d or 2d)
        lat   - latitude of mesh (cell corners) (1d or 2d)
        area  - area of cells (2d)
        lon0  - used when generating a spherical grid in absence of (lon,lat)
        is_geo_coord - If true, lon and lat are geographic coordinates,
                       Otherwise lon and lat are x and y from a map projection
        rfl   - refining level of this mesh
        """
        if (shape is None) and (lon is None) and (lat is None): raise Exception('Either shape must be specified or both lon and lat')
        if (lon is None) and (lat is not None): raise Exception('Either shape must be specified or both lon and lat')
        if (lon is not None) and (lat is None): raise Exception('Either shape must be specified or both lon and lat')
        # Determine shape
        if shape is not None:
            (nj,ni) = shape
        else: # Determine shape from lon and lat
            if (lon is None) or (lat is None): raise Exception('Either shape must be specified or both lon and lat')
            if len(lon.shape)==1: ni = lon.shape[0]-1
            elif len(lon.shape)==2: ni = lon.shape[1]-1
            else: raise Exception('lon must be 1D or 2D.')
            if len(lat.shape)==1 or len(lat.shape)==2: nj = lat.shape[0]-1
            else: raise Exception('lat must be 1D or 2D.')
        if from_cell_center: # Replace cell center coordinates with node coordinates
            ni,nj = ni+1, nj+1
            tmp = np.zeros(ni+1)
            tmp[1:-1] = 0.5 * ( lon[:-1] + lon[1:] )
            tmp[0] = 1.5 * lon[0] - 0.5 * lon[1]
            tmp[-1] = 1.5 * lon[-1] - 0.5 * lon[-2]
            lon = tmp
            tmp = np.zeros(nj+1)
            tmp[1:-1] = 0.5 * ( lat[:-1] + lat[1:] )
            tmp[0] = 1.5 * lat[0] - 0.5 * lat[1]
            tmp[-1] = 1.5 * lat[-1] - 0.5 * lat[-2]
            lat = tmp
        self.ni = ni
        self.nj = nj
        self.shape = (nj,ni)
        # Check shape of arrays and construct 2d coordinates
        if lon is not None and lat is not None:
            if len(lon.shape)==1:
                if len(lat.shape)>1: raise Exception('lon and lat must either be both 1d or both 2d')
                if lon.shape[0] != ni+1: raise Exception('lon has the wrong length')
            if len(lat.shape)==1:
                if len(lon.shape)>1: raise Exception('lon and lat must either be both 1d or both 2d')
                if lat.shape[0] != nj+1: raise Exception('lat has the wrong length')
            if len(lon.shape)==2 and len(lat.shape)==2:
                if lon.shape != lat.shape: raise Exception('lon and lat are 2d and must be the same size')
                if lon.shape != (nj+1,ni+1): raise Exception('lon has the wrong size')
                self.lon = lon
                self.lat = lat
            else:
                self.lon, self.lat = np.meshgrid(lon,lat)
        else: # Construct coordinates
            lon1d = np.linspace(-90.,90.,nj+1)
            lat1d = np.linspace(lon0,lon0+360.,ni+1)
            self.lon, self.lat = np.meshgrid(lon1d,lat1d)
        if area is not None:
            if area.shape != (nj,ni): raise Exception('area has the wrong shape or size')
            self.area = area
        else:
            self.area = None

        self.is_geo_coord = is_geo_coord
        if self.is_geo_coord:
            # Check and save North Pole point indices
            jj, ii = np.nonzero(self.lat==90)
            self.np_index = list(zip(jj, ii))

            # has_lon_jumps attribute is used to decide whether to use 2D interpolation with
            #   simple averages (fast) or shorter distance averages (slow).
            # Longitude will have jumps if:
            #   1. Difference in longitude value between neighboring points is larger than half the circle.
            #   2. Any of the North Pole points is not at the boundary, there must be jumps in longitude.
            self.has_lon_jumps = (max(np.abs(np.diff(self.lon, axis=0)).max(),
                                      np.abs(np.diff(self.lon, axis=1)).max()) > 180.0) or \
                                  np.any( (jj<self.nj) & (jj>0) & (ii<self.ni) & (ii>0) )

        self.rfl = rfl #refining level

    def __copy__(self):
        return GMesh(shape = self.shape, lon=self.lon, lat=self.lat, area=self.area)
    def copy(self):
        """Returns new instance with copied values"""
        return self.__copy__()
    def __repr__(self):
        return '<%s nj:%i ni:%i shape:(%i,%i)>'%(self.__class__.__name__,self.nj,self.ni,self.shape[0],self.shape[1])
    def __getitem__(self, key):
        return getattr(self, key)
    def transpose(self):
        """Transpose data swapping i-j indexes"""
        self.ni, self.nj = self.nj, self.ni
        self.shape = (self.nj, self.ni)
        self.lat, self.lon = self.lon.T, self.lat.T
        if self.area is not None: self.area = self.area.T
    def dump(self):
        """Dump Mesh to tty."""
        print(self)
        print('lon = ',self.lon)
        print('lat = ',self.lat)

    def plot(self, axis, subsample=1, linecolor='k', **kwargs):
        for i in range(0,self.ni+1,subsample):
            axis.plot(self.lon[:,i], self.lat[:,i], linecolor, **kwargs)
        for j in range(0,self.nj+1,subsample):
            axis.plot(self.lon[j,:], self.lat[j,:], linecolor, **kwargs)

    def pcolormesh(self, axis, data, **kwargs):
        return axis.pcolormesh( self.lon, self.lat, data, **kwargs)

    def __lonlat_to_XYZ(lon, lat):
        """Private method. Returns 3d coordinates (X,Y,Z) of spherical coordiantes (lon,lat)."""
        deg2rad = np.pi/180.
        lonr,latr = deg2rad*lon, deg2rad*lat
        return np.cos( latr )*np.cos( lonr ), np.cos( latr )*np.sin( lonr ), np.sin( latr )

    def __XYZ_to_lonlat(X, Y, Z):
        """Private method. Returns spherical coordinates (lon,lat) of 3d coordinates (X,Y,Z)."""
        rad2deg = 180./np.pi
        lat = np.arcsin( Z ) * rad2deg # -90 .. 90
        # Normalize X,Y to unit circle
        sub_roundoff = 2./np.finfo(X[0,0]).max
        R = 1. / ( np.sqrt(X*X + Y*Y) + sub_roundoff )
        lon = np.arccos( R*X ) * rad2deg # 0 .. 180
        lon = np.where( Y>=0, lon, -lon ) # Handle -180 .. 0
        return lon,lat

    def __mean2j(A):
        """Private method. Returns 2-point mean along j-direction."""
        return 0.5 * ( A[:-1,:] + A[1:,:] )

    def __mean2i(A):
        """Private method. Returns 2-point mean along i-direction."""
        return 0.5 * ( A[:,:-1] + A[:,1:] )

    def __mean4(A):
        """Private method. Returns 4-point mean (nodes to centers)."""
        return 0.25 * ( ( A[:-1,:-1] + A[1:,1:] ) + ( A[1:,:-1] + A[:-1,1:] ) )

    def __mean_from_xyz(X, Y, Z, direction):
        """Private method. Calculates means of (X,Y,Z) and converts to (lon,lat)."""
        # Refine mesh in 3d and project onto sphere
        if direction == 'j':
            X, Y, Z = GMesh.__mean2j(X), GMesh.__mean2j(Y), GMesh.__mean2j(Z)
        elif direction == 'i':
            X, Y, Z = GMesh.__mean2i(X), GMesh.__mean2i(Y), GMesh.__mean2i(Z)
        elif direction == '4':
            X, Y, Z = GMesh.__mean4(X), GMesh.__mean4(Y), GMesh.__mean4(Z)
        else:
            raise Exception('Wrong direction name')
        R = 1. / np.sqrt((X*X + Y*Y) + Z*Z)
        X,Y,Z = R*X, R*Y, R*Z

        # Normalize X,Y to unit circle
        #sub_roundoff = 2./np.finfo(X[0,0]).max
        #R = 1. / ( np.sqrt(X*X + Y*Y) + sub_roundoff )
        #X = R * X
        #Y = R * Y

        # Convert from 3d to spherical coordinates
        return GMesh.__XYZ_to_lonlat(X, Y, Z)

    def __lonmean2(lon1, lon2, period=360.0):
        """Private method. Returns 2-point mean for longitude with the consideration of periodicity. """
        # The special scenario that lon1 and lon2 are exactly 180-degree apart is unlikely to encounter here and
        # therefore ignored. In that scenario, 2D average is more complicated and requires latitude information.
        distance = np.mod(lon2-lon1, period)
        return lon1 + 0.5 * ( distance - (distance>0.5*period) * period )

    def __mean2j_lon(A, periodicity=True, singularities=[]):
        """Private method. Returns 2-point mean along j-direction for longitude.
        Singularities (if exists) appropriate their neighbor values.
        """
        if periodicity:
            mean_lon = GMesh.__lonmean2( A[:-1,:], A[1:,:] )
        else:
            mean_lon = GMesh.__mean2j(A)
        for jj, ii in singularities:
            if jj<A.shape[0]-1:
                mean_lon[jj, ii] = A[jj+1, ii]
            if jj>=1:
                mean_lon[jj-1, ii] = A[jj-1, ii]
        return mean_lon

    def __mean2i_lon(A, periodicity=True, singularities=[]):
        """Private method. Returns 2-point mean along i-direction for longitude.
        Singularities (if exists) appropriate their neighbor values.
        """
        if periodicity:
            mean_lon = GMesh.__lonmean2( A[:,:-1], A[:,1:] )
        else:
            mean_lon = GMesh.__mean2i(A)
        for jj, ii in singularities:
            #  A: i-1,   i,   i+1
            # mA:    i-1,   i
            if ii<A.shape[1]-1:
                mean_lon[jj, ii] = A[jj, ii+1]
            if ii>=1:
                mean_lon[jj, ii-1] = A[jj, ii-1]
        return mean_lon

    def __mean4_lon(A, periodicity=True, singularities=[]):
        """Private method. Returns 4-point mean (nodes to centers) for longitude.
        Singularities (if exists) appropriate their neighbor values.
        """
        if periodicity:
            mean_lon = GMesh.__lonmean2(GMesh.__lonmean2(A[:-1,:-1], A[1:,1:]),
                                        GMesh.__lonmean2(A[1:,:-1], A[:-1,1:]))
            for jj, ii in singularities:
                if jj<A.shape[0]-1 and ii<A.shape[1]-1:
                    mean_lon[jj, ii] = GMesh.__lonmean2(A[jj+1, ii+1], GMesh.__lonmean2(A[jj, ii+1], A[jj+1, ii]))
                if jj>=1 and ii>=1:
                    mean_lon[jj-1, ii-1] = GMesh.__lonmean2(A[jj-1, ii-1], GMesh.__lonmean2(A[jj, ii-1], A[jj-1, ii]))
                if jj<A.shape[0]-1 and ii>=1:
                    mean_lon[jj, ii-1] = GMesh.__lonmean2(A[jj+1, ii-1], GMesh.__lonmean2(A[jj, ii-1], A[jj+1, ii]))
                if jj>=1 and ii<A.shape[1]-1:
                    mean_lon[jj-1, ii] = GMesh.__lonmean2(A[jj-1, ii+1], GMesh.__lonmean2(A[jj, ii+1], A[jj-1, ii]))
        else:
            mean_lon = GMesh.__mean4(A)
            for jj, ii in singularities:
                if jj<A.shape[0]-1 and ii<A.shape[1]-1:
                    mean_lon[jj, ii] = 0.5 * A[jj+1, ii+1] + 0.25 * (A[jj, ii+1] + A[jj+1, ii])
                if jj>=1 and ii>=1:
                    mean_lon[jj-1, ii-1] = 0.5 * A[jj-1, ii-1] + 0.25 * (A[jj, ii-1] + A[jj-1, ii])
                if jj<A.shape[0]-1 and ii>=1:
                    mean_lon[jj, ii-1] = 0.5 * A[jj+1, ii-1] + 0.25 * (A[jj, ii-1] + A[jj+1, ii])
                if jj>=1 and ii<A.shape[1]-1:
                    mean_lon[jj-1, ii] = 0.5 * A[jj-1, ii+1] + 0.25 * (A[jj, ii+1] + A[jj-1, ii])
        return mean_lon

    def interp_center_coords(self, work_in_3d=True):
        """Returns interpolated center coordinates from nodes"""
        if self.is_geo_coord:
            if work_in_3d:
                # Calculate 3d coordinates of nodes (X,Y,Z), Z points along pole, Y=0 at lon=0,180, X=0 at lon=+-90
                X,Y,Z = GMesh.__lonlat_to_XYZ(self.lon, self.lat)
                lon, lat = GMesh.__mean_from_xyz(X, Y, Z, '4')
            else:
                lon, lat = GMesh.__mean4_lon(self.lon, periodicity=self.has_lon_jumps, singularities=self.np_index), GMesh.__mean4(self.lat)

        else:
            lon, lat = GMesh.__mean4(self.lon), GMesh.__mean4(self.lat)
        return lon, lat

    def refineby2(self, work_in_3d=True):
        """Returns new Mesh instance with twice the resolution"""
        lon, lat = np.zeros( (2*self.nj+1, 2*self.ni+1) ), np.zeros( (2*self.nj+1, 2*self.ni+1) )
        lon[::2,::2], lat[::2,::2] = self.lon, self.lat # Shared nodes
        if self.is_geo_coord:
            if work_in_3d:
                # Calculate 3d coordinates of nodes (X,Y,Z), Z points along pole, Y=0 at lon=0,180, X=0 at lon=+-90
                X,Y,Z = GMesh.__lonlat_to_XYZ(self.lon, self.lat)
                # lon[::2,::2], lat[::2,::2] = np.mod(self.lon+180.0, 360.0)-180.0, self.lat # only if we REALLY want the coords to be self-consistent
                lon[1::2,::2], lat[1::2,::2] = GMesh.__mean_from_xyz(X, Y, Z, 'j') # Mid-point along j-direction
                lon[::2,1::2], lat[::2,1::2] = GMesh.__mean_from_xyz(X, Y, Z, 'i') # Mid-point along i-direction
                lon[1::2,1::2], lat[1::2,1::2] = GMesh.__mean_from_xyz(X, Y, Z, '4') # Mid-point of cell
            else:
                lon[1::2,::2] = GMesh.__mean2j_lon(self.lon, periodicity=self.has_lon_jumps, singularities=self.np_index)
                lon[::2,1::2] = GMesh.__mean2i_lon(self.lon, periodicity=self.has_lon_jumps, singularities=self.np_index)
                lon[1::2,1::2] = GMesh.__mean4_lon(self.lon, periodicity=self.has_lon_jumps, singularities=self.np_index)
                lat[1::2,::2] = GMesh.__mean2j(self.lat)
                lat[::2,1::2] = GMesh.__mean2i(self.lat)
                lat[1::2,1::2] = GMesh.__mean4(self.lat)
        else:
            lon[1::2,::2] = GMesh.__mean2j(self.lon)
            lon[::2,1::2] = GMesh.__mean2i(self.lon)
            lon[1::2,1::2] = GMesh.__mean4(self.lon)
            lat[1::2,::2] = GMesh.__mean2j(self.lat)
            lat[::2,1::2] = GMesh.__mean2i(self.lat)
            lat[1::2,1::2] = GMesh.__mean4(self.lat)
        return GMesh(lon=lon, lat=lat, rfl=self.rfl+1, is_geo_coord=self.is_geo_coord)

    def max_spacings(self, masks=[]):
        """Returns the maximum spacing in lon and lat at each grid"""
        def mdist(x1, x2):
            """Returns positive distance modulo 360."""
            return np.minimum(np.mod(x1 - x2, 360.0), np.mod(x2 - x1, 360.0))
        lon, lat = self.lon, self.lat # aliasing
        # For each grid, find the largest spacing between any two of the four nodes
        dlat = np.max( np.stack( [np.abs(lat[:-1,:-1]-lat[:-1,1:]), np.abs(lat[1:,:-1]-lat[1:,1:]),
                                  np.abs(lat[:-1,:-1]-lat[1:,:-1]), np.abs(lat[1:,1:]-lat[:-1,1:]),
                                  np.abs(lat[:-1,:-1]-lat[1:,1:]), np.abs(lat[1:,:-1]-lat[:-1,1:])] ), axis=0)

        if self.is_geo_coord:
            dlon = np.max( np.stack( [mdist(lon[:-1,:-1], lon[:-1,1:]), mdist(lon[1:,:-1], lon[1:,1:]),
                                      mdist(lon[:-1,:-1], lon[1:,:-1]), mdist(lon[1:,1:], lon[:-1,1:]),
                                      mdist(lon[:-1,:-1], lon[1:,1:]), mdist(lon[1:,:-1], lon[:-1,1:])] ), axis=0)
            # Treat the ambiguity of the North Pole longitude
            for jj, ii in self.np_index:
                if jj<self.nj and ii<self.ni:
                    dlon[jj,ii] = np.max( [mdist(lon[jj+1,ii+1], lon[jj,ii+1]), mdist(lon[jj+1,ii+1], lon[jj+1,ii]),
                                        mdist(lon[jj,ii+1], lon[jj+1,ii])] )
                if jj>=1 and ii>=1:
                    dlon[jj-1,ii-1] = np.max( [mdist(lon[jj-1,ii-1], lon[jj,ii-1]), mdist(lon[jj-1,ii-1], lon[jj-1,ii]),
                                            mdist(lon[jj,ii-1], lon[jj-1,ii])] )
                if jj<self.nj and ii>=1:
                    dlon[jj,ii-1] = np.max( [mdist(lon[jj+1,ii-1], lon[jj,ii-1]), mdist(lon[jj+1,ii-1], lon[jj+1,ii]),
                                            mdist(lon[jj,ii-1], lon[jj+1,ii])] )
                if jj>=1 and ii<self.ni:
                    dlon[jj-1,ii] = np.max( [mdist(lon[jj-1,ii+1], lon[jj,ii+1]), mdist(lon[jj-1,ii+1], lon[jj-1,ii]),
                                            mdist(lon[jj,ii+1], lon[jj-1,ii])] )
        else:
            dlon = np.max( np.stack( [np.abs(lon[:-1,:-1]-lon[:-1,1:]), np.abs(lon[1:,:-1]-lon[1:,1:]),
                                      np.abs(lon[:-1,:-1]-lon[1:,:-1]), np.abs(lon[1:,1:]-lon[:-1,1:]),
                                      np.abs(lon[:-1,:-1]-lon[1:,1:]), np.abs(lon[1:,:-1]-lon[:-1,1:])] ), axis=0)

        # Mask out rectangles
        for Js, Je, Is, Ie in masks:
            jst, jed, ist, ied = Js*(2**self.rfl), Je*(2**self.rfl), Is*(2**self.rfl), Ie*(2**self.rfl)
            dlon[jst:jed, ist:ied], dlat[jst:jed, ist:ied] = 0.0, 0.0
        return dlon, dlat

    def max_refine_levels(dlon_tgt, dlat_tgt, dlon_src, dlat_src):
        """Return the maximum refine levels needed based on given target and source data resolutions"""
        return np.maximum( np.ceil( np.log2( dlat_tgt/dlat_src ) ),
                           np.ceil( np.log2( dlon_tgt/dlon_src ) ) )

    def rotate(self, y_rot=0, z_rot=0):
        """Sequentially apply a rotation about the Y-axis and then the Z-axis."""
        deg2rad = np.pi/180.
        # Calculate 3d coordinates of nodes (X,Y,Z), Z points along pole, Y=0 at lon=0,180, X=0 at lon=+-90
        X,Y,Z = GMesh.__lonlat_to_XYZ(self.lon, self.lat)
        # Rotate anti-clockwise about Y-axis
        C,S = np.cos( deg2rad*y_rot ), np.sin( deg2rad*y_rot )
        X,Z = C*X + S*Z, -S*X + C*Z
        # Rotate anti-clockwise about Y-axis
        C,S = np.cos( deg2rad*z_rot ), np.sin( deg2rad*z_rot )
        X,Y = C*X - S*Y, S*X + C*Y

        # Convert from 3d to spherical coordinates
        self.lon,self.lat = GMesh.__XYZ_to_lonlat(X, Y, Z)

        return self

    def coarsenby2(self, coarser_mesh, timers=False):
        """Set the height for lower level Mesh by coarsening"""
        if(self.rfl == 0):
            raise Exception('Coarsest grid, no more coarsening possible!')

        if timers: gtic = GMesh._toc(None, "")
        coarser_mesh.height = 0.25 * ( ( self.height[:-1:2,:-1:2] + self.height[1::2,1::2] )
                                     + ( self.height[1::2,:-1:2] + self.height[:-1:2,1::2] ) )
        if timers: gtic = GMesh._toc(gtic, "Whole process")

    def find_nn_uniform_source(self, eds, use_center=True, work_in_3d=True, debug=False):
        """Returns the i,j arrays for the indexes of the nearest neighbor centers at (lon,lat) to the self nodes
        The option use_center=True is default so that lon,lat are cell-center coordinates."""

        if use_center:
            # Searching for source cells that the self centers fall into
            lon_tgt, lat_tgt = self.interp_center_coords(work_in_3d=work_in_3d)
        else:
            # Searching for source cells that the self nodes fall into
            lon_tgt, lat_tgt = self.lon, self.lat
        nn_i,nn_j = eds.indices( lon_tgt, lat_tgt )
        if debug:
            print('Self lon =',eds.lonh[0],'...',eds.lonh[-1])
            print('Self lat =',eds.lath[0],'...',eds.lath[-1])
            print('Target lon =',lon_tgt)
            print('Target lat =',lat_tgt)
            print('Source lon =',eds.lonh[nn_i])
            print('Source lat =',eds.lath[nn_j])
            print('NN i =',nn_i)
            print('NN j =',nn_j)
        assert nn_j.min()>=0, 'Negative j index calculated! j='+str(nn_j.min())
        assert nn_j.max()<eds.nj, 'Out of bounds j index calculated! j='+str(nn_j.max())
        assert nn_i.min()>=0, 'Negative i index calculated! i='+str(nn_i.min())
        assert nn_i.max()<eds.ni, 'Out of bounds i index calculated! i='+str(nn_i.max())
        return nn_i,nn_j

    def source_hits(self, eds, use_center=True, work_in_3d=True, singularity_radius=0.25):
        """Returns an mask array of 1's if a cell with center (xs,ys) is intercepted by a node
           on the mesh, 0 if no node falls in a cell"""
        # Indexes of nearest xs,ys to each node on the mesh
        i,j = self.find_nn_uniform_source(eds, use_center=use_center, work_in_3d=work_in_3d)
        hits = np.zeros((eds.nj, eds.ni))
        if self.is_geo_coord and singularity_radius>0:
            hits[np.abs(eds.lath)>90-singularity_radius,:] = 1 # use indices instead to avoid alloccation of lath
        hits[j,i] = 1
        return hits

    def _toc(tic, label):
        if tic is not None:
            dt = ( time.time_ns() - tic ) // 1000000
            if dt<9000: print( '{:>10}ms : {}'.format( dt, label) )
            else: print( '{:>10}secs : {}'.format( dt / 1000, label) )
        return time.time_ns()

    def refine_loop(self, eds, max_stages=32, max_mb=32000, fixed_refine_level=0, resolution_limit=True, mask_res=[],
                    work_in_3d=True, use_center=True, singularity_radius=0.25, verbose=True, timers=False):
        """Repeatedly refines the mesh until all cells in the source grid are intercepted by mesh nodes.
           Returns a list of the refined meshes starting with parent mesh.
        Level of refinement is decided in the following order:
        1) If fixed_refine_level is specified, the refine loop will stop after reaching the specified level
        2) If resolution_limit is switched on, the refine loop will stop after reaching a pre-evaluated level based on target
        grid resolution. This avoids unnessary refinement when the target grid boundaries are not a straight lon/lat lines.
        3) Otherwise, the refine loop will stop whichever the following conditions is satisified first
            A) All cells are intercepted or nor more cells are intercepted.
            B) Memory (max_mb) or stage (max_stages) limit is reached.
        """
        if verbose: print(self)
        if timers: gtic = GMesh._toc(None, "")
        GMesh_list, this = [self], self
        mb = 2*8*this.shape[0]*this.shape[1]/1024/1024
        converged = False
        if fixed_refine_level>0:
            resolution_limit = False # fixed_refine_level overrides resolution_limit
            max_rfl = fixed_refine_level
        elif resolution_limit:
            dellon_t, dellat_t = self.max_spacings(masks=mask_res)
            max_rfl = np.max( GMesh.max_refine_levels(dellon_t, dellat_t, *eds.spacing()) ).astype(int)
            if np.ma.is_masked(max_rfl): max_rfl = 0 # For dellon_t=0 and dellat_t=0 (fully masked domain)
        else:
            # Equivalent to prior loop-breaking conditions: len(GMesh_list)==max_stages or 4*mb>=max_mb
            max_rfl = min( max_stages-1, np.floor(np.log2(max_mb/mb)*0.5).astype(int) )
            hits = this.source_hits(eds, use_center=use_center, singularity_radius=singularity_radius)
            nhits, prev_hits = hits.sum().astype(int), 0
            converged = np.all(hits) or (nhits==prev_hits)
        if timers: tic = GMesh._toc(gtic, "Set up")
        if verbose:
            print('Refine level', this.rfl, repr(this), end=" ")
            if not (resolution_limit or (fixed_refine_level>0)):
                print('Hit', nhits, 'out of', hits.size, 'cells', end=" ")
            if resolution_limit:
                dellon_tm, dellat_tm = dellon_t.max(), dellat_t.max()
                spc_lon = int(1/dellon_tm) if dellon_tm!=0 else float('Inf')
                spc_lat = int(1/dellat_tm) if dellat_tm!=0 else float('Inf')
                print('dx~1/{} dy~1/{}: refine levels needs={}'.format(spc_lon, spc_lat, max_rfl), end=" ")
            print('(%.4f'%mb,'Mb)')

        for _ in range(1, max_rfl+1):
            this = this.refineby2(work_in_3d=work_in_3d)
            if timers: stic = GMesh._toc(tic, "refine by 2")
            if not (resolution_limit or (fixed_refine_level>0)):
                hits = this.source_hits(eds, singularity_radius=singularity_radius)
                if timers: stic = GMesh._toc(stic, "calculate hits on topo grid")
                nhits, prev_hits = hits.sum().astype(int), nhits
                converged = converged or np.all(hits) or (nhits==prev_hits)
            if not converged: GMesh_list.append( this )
            if timers: stic = GMesh._toc(stic, "extending list")
            if timers: tic = GMesh._toc(tic, "Total for loop")
            if verbose:
                print('Refine level', this.rfl, repr(this), end=" ")
                if not (resolution_limit or (fixed_refine_level>0)):
                    print('Hit', nhits, 'out of', hits.size, 'cells', end=" ")
                mb = mb * 4
                print('(%.4f'%mb,'Mb)')
            if converged: break

        if (not converged) and (not (resolution_limit or (fixed_refine_level>0))):
            print("Warning: Maximum number of allowed refinements reached without all source cells hit.")
        if timers: tic = GMesh._toc(gtic, "Total for whole process")

        return GMesh_list

    def project_source_data_onto_target_mesh(self, eds, use_center=True, work_in_3d=True, timers=False):
        """Returns the EDS data on the target mesh (self) with values equal to the nearest-neighbor source point data"""
        if timers: gtic = GMesh._toc(None, "")
        if use_center:
            self.height = np.zeros((self.nj,self.ni))
            tx, ty = self.interp_center_coords(work_in_3d=work_in_3d)
        else:
            self.height = np.zeros((self.nj+1,self.ni+1))
            tx, ty = self.lon, self.lat
        if timers: tic = GMesh._toc(gtic, "Allocate memory")
        nns_i, nns_j = eds.indices( tx, ty )
        if timers: tic = GMesh._toc(tic, "Calculate interpolation indexes")
        self.height[:,:] = eds.data[nns_j[:,:], nns_i[:,:]]
        if timers: tic = GMesh._toc(tic, "indirect indexing")
        if timers: tic = GMesh._toc(gtic, "Whole process")

class RegularCoord:
    """Container for uniformly spaced global cell center coordinate parameters

    For use with uniformly gridded data that has cell center global coordinates"""
    def __init__( self, n, origin, periodic, delta=None, degppi=180 ):
        """Create a RegularCoord
        n         is number of cells;
        origin    is the coordinate on the left edge (not first);
        periodic  distinguishes between longitude and latitude
        """
        self.n = n # Global parameter
        self.periodic = periodic # Global parameter
        if delta is not None:
            self.delta, self.rdelta = delta, 1.0/delta
        else:
            if periodic: self.delta, self.rdelta = ( 2 * degppi ) / n, n / ( 2 * degppi )  # Global parameter
            else: self.delta, self.rdelta = degppi / n, n / degppi # Global parameter
        self.origin = origin # Global parameter
        self.offset = np.floor( self.rdelta * self.origin ).astype(int) # Global parameter
        self.rem = np.mod( self.rdelta * self.origin, 1 ) # Global parameter ( needed for odd n)
        self.start = 0 # Special for each subset
        self.stop = self.n # Special for each subset
        self._centers, self._bounds = None, None
    def __repr__( self ):
        return '<RegularCoord n={}, dx={}, rdx={}, x0={}, io={}, rem={}, is-ie={}-{}, periodic={}>'.format( \
            self.n, self.delta, self.rdelta, self.origin, self.offset, self.rem, self.start, self.stop, self.periodic)
    @property
    def size(self):
        """Return the size of the coordinate"""
        return self.stop - self.start + self.n * int( self.periodic and self.start>=self.stop )
    @property
    def centers(self):
        """Return center coordinates (length = size)"""
        if self._centers is None or self._centers.size!=self.size:
            if self.periodic and self.start>=self.stop:
                self._centers = self.origin + self.delta * np.r_[np.arange(self.start+0.5, self.n),
                                                                 np.arange(self.n+0.5, self.n+self.stop)]
            else:
                self._centers = self.origin + self.delta * np.arange(self.start+0.5, self.stop)
        return self._centers
    @property
    def bounds(self):
        """Return boundary coordinates (length = size+1)"""
        if self._bounds is None or self._bounds.size!=self.size+1:
            if self.periodic and self.start>=self.stop:
                self._bounds = self.origin + self.delta * np.r_[np.arange(self.start, self.n),
                                                                np.arange(self.n, self.n+self.stop+1)]
            else:
                self._bounds = self.origin + self.delta * np.arange(self.start, self.stop+1)
        return self._bounds
    def subset( self, start=None, stop=None ):
        """Subset a RegularCoord with slice "slc" """
        Is, Ie = 0, self.n
        if start is not None: Is = start
        if stop is not None: Ie = stop
        assert (Is<Ie and (not self.periodic)) or self.periodic, "start is larger than stop in non-periodic coordinate."
        if Is==Ie and self.periodic: # Only happens when all longitudes are included and shifting origin is likely unnecessary.
            Is, Ie = 0, self.n
        S = RegularCoord( self.n, self.origin, self.periodic, delta=self.delta ) # This creates a copy of "self"
        S.start, S.stop = Is, Ie
        return S
    def indices( self, x, bound_subset=False ):
        """Return indices of cells that contain x

        If RegularCoord is non-periodic (i.e. latitude), out of range values of "x" will be clipped to -90..90 .
        If regularCoord is periodic, any value of x will be globally wrapped.
        If RegularCoord is a subset, then "x" will be clipped to the bounds of the subset (after periodic wrapping).
        if "bound_subset" is True, then limit indices to the range of the subset
        """
        # number of grid points from origin (global first edge)
        ind = np.floor( self.rdelta * np.array(x) - self.rem ).astype(int) - self.offset
        # Apply global bounds and reference to start
        if self.periodic:
            ind = np.mod( ind - self.start, self.n )
        else:
            ind = np.maximum( 0, np.minimum( self.n - 1, ind ) ) - self.start
        assert ind.min() >= 0, "out of range"
        assert ind.max() < self.n, "out of range"
        # Now adjust for subset
        if bound_subset:
            ind = np.maximum( 0, np.minimum( self.size - 1, ind ) )
            assert ind.min() >= 0, "out of range"
            assert ind.max() < self.size, "out of range"
        return ind

class UniformEDS:
    """Container for a uniform elevation dataset"""
    def __init__( self, lon=None, lat=None, elevation=None ):
        """(lon,lat) are cell centers and 1D with combined shape equalt that of elevation."""
        if elevation is None: # When creating a subset, we temporarily allow the creation of a "blank" UniformEDS
            self.lon_coord, self.lat_coord = None, None
            self.data = np.zeros((0))
        else: # This is the real constructor for a gloal domain
            assert len(lon.shape) == 1, "Longitude must be 1D"
            assert len(lat.shape) == 1, "Latitude must be 1D"
            assert len(lon) == elevation.shape[1], "Inconsistent longitude shape"
            assert len(lat) == elevation.shape[0], "Inconsistent latitude shape"
            ni, nj = len(lon), len(lat)
            dlon, dlat = 360. / ni, 180 / nj
            assert np.abs( lon[-1] - lon[0] - 360 + dlon ) < 1.e-5 * dlon, "longitude does not appear to be global"
            assert np.abs( lat[-1] - lat[0] - 180 + dlat ) < 1.e-5 * dlat, "latitude does not appear to be global"
            lon0 = np.floor( lon[0] - 0.5 * dlon + 0.5 ) # Calculating the phase this way restricts ourselves to data starting on integer values
            assert np.abs( lon[0] - 0.5 * dlon - lon0 ) < 1.e-9 * dlon, "edge of longitude is not a round number"
            assert np.abs( lat[0] - 0.5 * dlat + 90 ) < 1.e-9 * dlat, "edge of latitude is not 90"
            self.lon_coord = RegularCoord( ni, lon0, True)
            self.lat_coord = RegularCoord( nj, -90, False)
            self.data = elevation
    def __repr__( self ):
        mem = ( self.ni * self.nj + self.ni + self.nj ) * 8 / 1024 / 1024 / 1024 # Gb
        return '<UniformEDS {} x {} ({:.3f}Gb)\nlon = {}\nh:{}\nq:{}\nlat = {}\nh:{}\nq:{}\ndata = {}>'.format( \
            self.ni, self.nj, mem, self.lon_coord, self.lonh, self.lonq, self.lat_coord, self.lath, self.latq, self.data.shape )
    @property
    def ni( self ):
        """Aliasing longitude length"""
        return self.lon_coord.size
    @property
    def nj( self ):
        """Aliasing latitude length"""
        return self.lat_coord.size
    @property
    def dlon( self ):
        """Aliasing longitude spacing"""
        return self.lon_coord.delta
    @property
    def dlat( self ):
        """Aliasing latitude spacing"""
        return self.lat_coord.delta
    @property
    def lonq( self ):
        """Aliasing longitude bounds"""
        return self.lon_coord.bounds
    @property
    def latq( self ):
        """Aliasing latitude bounds"""
        return self.lat_coord.bounds
    @property
    def lonh( self ):
        """Aliasing longitude centers"""
        return self.lon_coord.centers
    @property
    def lath( self ):
        """Aliasing latitude centers"""
        return self.lat_coord.centers
    def spacing( self ):
        """Returns the longitude and latitude spacing"""
        return self.dlon, self.dlat
    def subset( self, Is, Ie, Js, Je ):
        """Subset a UniformEDS as [jslice,islice]"""
        S = UniformEDS()
        S.lon_coord = self.lon_coord.subset( start=Is, stop=Ie )
        S.lat_coord = self.lat_coord.subset( start=Js, stop=Je )
        if Is>Ie:
            S.data = np.c_[ self.data[Js:Je, Is:], self.data[Js:Je, :Ie] ]
        else:
            S.data = self.data[Js:Je, Is:Ie]
        return S
    def indices( self, lon, lat, bound_subset=False ):
        """Return the i,j indices of cells in which (lon,lat) fall"""
        return self.lon_coord.indices( lon, bound_subset=bound_subset ), self.lat_coord.indices( lat, bound_subset=bound_subset )
    def bb_slices( self, lon, lat, halo_lon=0, halo_lat=0 ):
        """Returns the slices defining the bounding box of data hit by (lon,lat)"""
        # si, sj = self.indices( lon, lat )
        # return slice( si.min(), si.max() +1 ), slice( sj.min(), sj.max() + 1 )
        loni, lati = self.lon_coord, self.lat_coord
        Is, Ie = np.mod(loni.indices( lon.min() ) - halo_lon, self.ni), np.mod(loni.indices( lon.max() ) + halo_lon + 1, self.ni)
        Js, Je = lati.indices( lat.min() ) - halo_lat, lati.indices( lat.max() ) + halo_lat + 1
        if Is+1==Ie: Is, Ie = 0, self.ni # All longitudes are included.
        return Is, Ie, Js, Je
    def plot(self, axis, subsample=None, **kwargs):
        if subsample is None:
            return axis.pcolormesh( self.lonq, self.latq, self.data, **kwargs )
        return axis.pcolormesh( self.lonq[::subsample], self.latq[::subsample], self.data[::subsample,::subsample], **kwargs )
