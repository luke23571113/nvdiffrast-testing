from nvdiffmodeling.src import obj
from nvdiffmodeling.src import mesh

load_mesh = obj.load_obj("chair.obj")

load_mesh = mesh.unit_size(load_mesh) #I think this normalizes it??


#need to normalize first ...  