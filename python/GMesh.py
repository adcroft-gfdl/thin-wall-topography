#!/usr/bin/env python

import numpy as np
import time

def is_mesh_uniform(lon,lat):
    """Returns True if the input grid (lon,lat) is uniform and False otherwise"""
    def compare(array):
        eps = np.finfo( array.dtype ).eps # Precision of datatype
        delta = np.abs( array[1:] - array[:-1] ) # Difference along first axis
        error = np.abs( array )
        error = np.maximum( error[1:], error[:-1] ) # Error in difference
        derror = np.abs( delta - delta.flatten()[0] ) # Tolerance to which comparison can be made
        return np.all( derror < ( error + error.flatten()[0] ) )
    assert len(lon.shape) == len(lat.shape), "Arguments lon and lat must have the same rank"
    if len(lon.shape)==2: # 2D arralat
        assert lon.shape == lat.shape, "Arguments lon and lat must have the same shape"
    if len(lon.shape)>2 or len(lat.shape)>2:
        raise Exception("Arguments must be either both be 1D or both be 2D arralat")
    return compare(lat) and compare(lon.T)

class IntCoord(object):
    """
    A type for integerized coordinate

    start : float
        Global starting lon/lat
    delta : float
        Resolution (in deg)
    N : int
        Total number of grid point
    n0 : int, optional
        Starting index of the subset
    n1 : int, optional
        Ending index of the subset
    """
    def __init__(self, start, delta, N, n0=0, n1=None):
        self.start = start
        self.delta = delta
        self.N = N
        self.n0 = n0
        self.n1 = n1
        if n1 is None:
            self.n1 = self.N
        self.size = self.n1 - self.n0 + self.N * int( self.n0>self.n1 )
        self._centers, self._bounds = None, None
    @property
    def centers(self):
        if self._centers is None:
            if self.n0>self.n1:
                self._centers = self.start + self.delta * np.r_[np.arange(self.n0, self.N),
                                                                np.arange(self.N, self.N+self.n1)]
            else:
                self._centers = self.start + self.delta * np.arange(self.n0, self.n1)
        return self._centers
    @property
    def bounds(self):
        if self._bounds is None:
            if self.n0>self.n1:
                self._bounds = self.start + self.delta * np.r_[np.arange(self.n0-0.5, self.N),
                                                               np.arange(self.N+0.5, self.N+self.n1)]
            else:
                self._bounds = self.start + self.delta * np.arange(self.n0-0.5, self.n1)
        return self._bounds

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

    def __init__(self, shape=None, lon=None, lat=None, area=None, lon0=-180., from_cell_center=False, rfl=0):
        """Constructor for Mesh:
        shape - shape of cell array, (nj,ni)
        ni    - number of cells in i-direction (last index)
        nj    - number of cells in j-direction (first index)
        lon   - longitude of mesh (cell corners) (1d or 2d)
        lat   - latitude of mesh (cell corners) (1d or 2d)
        area  - area of cells (2d)
        lon0  - used when generating a spherical grid in absence of (lon,lat)
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

        self.rfl = rfl #refining level

    # def read_center_coords(self, lont, latt):
    #     if lont.shape != (self.nj,self.ni): raise Exception('Cell center lon has the wrong size')
    #     if latt.shape != (self.nj,self.ni): raise Exception('Cell center lat has the wrong size')
    #     self.lont, self.latt = lont, latt

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

    def interp_center_coords(self, work_in_3d=True):
        """Returns interpolated center coordinates from nodes"""
        if work_in_3d:
            # Calculate 3d coordinates of nodes (X,Y,Z), Z points along pole, Y=0 at lon=0,180, X=0 at lon=+-90
            X,Y,Z = GMesh.__lonlat_to_XYZ(self.lon, self.lat)
            lon, lat = GMesh.__mean_from_xyz(X, Y, Z, '4')
        else:
            lon, lat = GMesh.__mean4(self.lon), GMesh.__mean4(self.lat)
        return lon, lat

    def refineby2(self, work_in_3d=True):
        """Returns new Mesh instance with twice the resolution"""
        lon, lat = np.zeros( (2*self.nj+1, 2*self.ni+1) ), np.zeros( (2*self.nj+1, 2*self.ni+1) )
        lon[::2,::2], lat[::2,::2] = self.lon, self.lat # Shared nodes
        if work_in_3d:
            # Calculate 3d coordinates of nodes (X,Y,Z), Z points along pole, Y=0 at lon=0,180, X=0 at lon=+-90
            X,Y,Z = GMesh.__lonlat_to_XYZ(self.lon, self.lat)
            # lon[::2,::2], lat[::2,::2] = np.mod(self.lon+180.0, 360.0)-180.0, self.lat # only if we REALLY want the coords to be self-consistent
            lon[1::2,::2], lat[1::2,::2] = GMesh.__mean_from_xyz(X, Y, Z, 'j') # Mid-point along j-direction
            lon[::2,1::2], lat[::2,1::2] = GMesh.__mean_from_xyz(X, Y, Z, 'i') # Mid-point along i-direction
            lon[1::2,1::2], lat[1::2,1::2] = GMesh.__mean_from_xyz(X, Y, Z, '4') # Mid-point of cell
        else:
            lon[1::2,::2], lat[1::2,::2] = GMesh.__mean2j(self.lon), GMesh.__mean2j(self.lat) # Mid-point along j-direction
            lon[::2,1::2], lat[::2,1::2] = GMesh.__mean2i(self.lon), GMesh.__mean2i(self.lat) # Mid-point along i-direction
            lon[1::2,1::2], lat[1::2,1::2] = GMesh.__mean4(self.lon), GMesh.__mean4(self.lat) # Mid-point of cell

        return GMesh(lon=lon, lat=lat, rfl=self.rfl+1)

    def coarsest_resolution(self, mask_idx=[]):
        """Returns the coarsest resolution at each grid"""
        def mdist(x1, x2):
            """Returns positive distance modulo 360."""
            return np.minimum(np.mod(x1 - x2, 360.0), np.mod(x2 - x1, 360.0))
        l, p = self.lon, self.lat
        del_lam = np.maximum(np.maximum(np.maximum(mdist(l[:-1,:-1], l[:-1,1:]), mdist(l[1:,:-1], l[1:,1:])),
                                        np.maximum(mdist(l[:-1,:-1], l[1:,:-1]), mdist(l[1:,1:], l[:-1,1:]))),
                             np.maximum(mdist(l[:-1,:-1], l[1:,1:]), mdist(l[1:,:-1], l[:-1,1:])))
        del_phi = np.maximum(np.maximum(np.maximum(np.abs(np.diff(p, axis=0))[:,1:], np.abs((np.diff(p, axis=0))[:,:-1])),
                                        np.maximum(np.abs(np.diff(p, axis=1))[1:,:], np.abs((np.diff(p, axis=1))[:-1,:]))),
                             np.maximum(np.abs(p[:-1,:-1]-p[1:,1:]), np.abs(p[1:,:-1]-p[:-1,1:])))
        if len(mask_idx)>0:
            for Js, Je, Is, Ie in mask_idx:
                jst, jed, ist, ied = Js*(2**self.rfl), Je*(2**self.rfl), Is*(2**self.rfl), Ie*(2**self.rfl)
                del_lam[jst:jed, ist:ied], del_phi[jst:jed, ist:ied] = 0.0, 0.0
        return del_lam, del_phi

    def max_refine_level(self, dlon_src, dlat_src):
        dlat, dlon = self.coarsest_resolution()
        # dlat_src, dlon_src = lon.delta, lat.delta

        return np.maximum( np.ceil( np.log2( dlat/dlat_src ) ),
                           np.ceil( np.log2( dlon/dlon_src ) ) )

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

    def coarsenby2(self, coarser_mesh, debug=False, timers=False):
        """Set the height for lower level Mesh by coarsening"""
        if(self.rfl == 0):
            raise Exception('Coarsest grid, no more coarsening possible!')

        if timers: gtic = GMesh._toc(None, "")
        coarser_mesh.height = 0.25 * ( ( self.height[:-1:2,:-1:2] + self.height[1::2,1::2] )
                                     + ( self.height[1::2,:-1:2] + self.height[:-1:2,1::2] ) )
        if timers: gtic = GMesh._toc(gtic, "Whole process")

    def find_nn_uniform_source(self, lon, lat, use_center=False):
        """Returns the i,j arrays for the indexes of the nearest neighbor point to grid (lon,lat)"""
        sni,snj = lon.N,lat.N # Shape of source
        # Spacing on uniform mesh
        dellon, dellat = lon.delta, lat.delta
        assert self.lat.max()<=lat.start+(lat.n1+0.5)*lat.delta, 'Mesh has latitudes above range of regular grid '+str(self.lat.max())+' '+str(lat.start+(lat.n1+0.5)*lat.delta)
        assert self.lat.min()>=lat.start+(lat.n0-0.5)*lat.delta, 'Mesh has latitudes below range of regular grid '+str(self.lat.min())+' '+str(lat.start+(lat.n0-0.5)*lat.delta)
        if use_center:
            lon_tgt, lat_tgt = self.interp_center_coords(work_in_3d=True)
        else:
            lon_tgt, lat_tgt = self.lon, self.lat
        # Nearest integer (the upper one if equidistant)
        nn_i = np.floor(np.mod(lon_tgt-lon.start+0.5*dellon,360)/dellon)
        nn_j = np.floor(0.5+(lat_tgt-lat.start)/dellat)
        nn_j = np.minimum(nn_j, snj-1)
        assert nn_j.min()>=0, 'Negative j index calculated! j='+str(nn_j.min())
        assert nn_j.max()<snj, 'Out of bounds j index calculated! j='+str(nn_j.max())
        assert nn_i.min()>=0, 'Negative i index calculated! i='+str(nn_i.min())
        assert nn_i.max()<sni, 'Out of bounds i index calculated! i='+str(nn_i.max())
        return nn_i.astype(int),nn_j.astype(int)

    def source_hits(self, xs, ys, use_center=False, singularity_radius=0.25):
        """Returns an mask array of 1's if a cell with center (xs,ys) is intercepted by a node
           on the mesh, 0 if no node falls in a cell"""
        # Indexes of nearest xs,ys to each node on the mesh
        i,j = self.find_nn_uniform_source(xs,ys,use_center=use_center)
        sni, snj = xs.size, ys.size # Shape of source
        hits = np.zeros((snj,sni))
        if singularity_radius>0:
            iy = (np.ceil((90-singularity_radius-ys.start)/ys.delta)-ys.n0).astype(int)
            hits[iy:] = 1
        hits[j-ys.n0, np.mod(i-xs.n0, xs.N)] = 1
        return hits

    def _toc(tic, label):
        if tic is not None:
            dt = ( time.time_ns() - tic ) // 1000000
            if dt<9000: print( '{:>10}ms : {}'.format( dt, label) )
            else: print( '{:>10}secs : {}'.format( dt / 1000, label) )
        return time.time_ns()

    def refine_loop(self, src_lon, src_lat, max_stages=32, max_mb=2000, fixed_refine_level=-1, work_in_3d=True,
                    use_center=False, resolution_limit=False, mask_res=[], singularity_radius=0.25, verbose=True, timers=False):
        """Repeatedly refines the mesh until all cells in the source grid are intercepted by mesh nodes.
           Returns a list of the refined meshes starting with parent mesh."""
        if timers: gtic = GMesh._toc(None, "")
        GMesh_list, this = [self], self
        hits = this.source_hits(src_lon, src_lat, use_center=use_center, singularity_radius=singularity_radius)
        nhits, prev_hits, mb = hits.sum().astype(int), 0, 2*8*this.shape[0]*this.shape[1]/1024/1024
        if verbose: print('Refine level', this.rfl, repr(this), 'Hit', nhits, 'out of', hits.size, 'cells (%.4f'%mb,'Mb)')
        fine = False
        if resolution_limit:
            dellon_s, dellat_s = src_lon.delta,src_lat.delta
            del_lam, del_phi = this.coarsest_resolution(mask_idx=mask_res)
            dellon_t, dellat_t = del_lam.max(), del_phi.max()
            fine = (dellon_t<=dellon_s) and (dellat_t<=dellat_s)
        converged = np.all(hits) or (nhits==prev_hits) or (resolution_limit and fine)
        if timers: tic = GMesh._toc(gtic, "Set up")
        # Conditions to refine
        # 1) Not all cells are intercepted
        # 2) A refinement intercepted more cells
        # 3) [if resolution_limit] Coarsest resolution in each direction is coarser than source.
        #    This avoids the excessive refinement which is essentially extrapolation.
        while(((not converged) and (len(GMesh_list)<max_stages) and (4*mb<max_mb) and (fixed_refine_level==-1)) or (this.rfl<fixed_refine_level)):
            if timers: tic = GMesh._toc(None, "")
            this = this.refineby2(work_in_3d=work_in_3d)
            if timers: stic = GMesh._toc(tic, "refine by 2")
            hits = this.source_hits(src_lon, src_lat, singularity_radius=singularity_radius)
            nhits, prev_hits, mb = hits.sum().astype(int), nhits, 2*8*this.shape[0]*this.shape[1]/1024/1024
            if resolution_limit:
                del_lam, del_phi = this.coarsest_resolution(mask_idx=mask_res)
                dellon_t, dellat_t = del_lam.max(), del_phi.max()
                fine = (dellon_t<=dellon_s) and (dellat_t<=dellat_s)
                if timers: stic = GMesh._toc(stic, "calculate resolution stopping criteria")
            converged = np.all(hits) or (nhits==prev_hits) or (resolution_limit and fine)
            if nhits>prev_hits or this.rfl<=fixed_refine_level:
                GMesh_list.append( this )
                if timers: stic = GMesh._toc(stic, "extending list")
                if timers: tic = GMesh._toc(tic, "Total for loop")
            if verbose: print('Refine level', this.rfl, this, 'Hit', nhits, 'out of', hits.size, 'cells (%.4f'%mb,'Mb)')

        if not converged:
            print("Warning: Maximum number of allowed refinements reached without all source cells hit.")
        if timers: tic = GMesh._toc(gtic, "Total for whole process")

        return GMesh_list

    def project_source_data_onto_target_mesh(self,xs,ys,zs,use_center=False):
        """Returns the array on target mesh with values equal to the nearest-neighbor source point data"""
        # if xs.shape != ys.shape: raise Exception('xs and ys must be the same shape')
        nns_i,nns_j = self.find_nn_uniform_source(xs,ys,use_center=use_center)
        if use_center:
            self.height = np.zeros((self.nj,self.ni))
        else:
            self.height = np.zeros((self.nj+1,self.ni+1))
        self.height[:,:] = zs[nns_j[:,:]-ys.n0, np.mod(nns_i[:,:]-xs.n0, xs.N)]
        return
