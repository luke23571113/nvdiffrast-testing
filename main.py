from nvdiffmodeling.src import obj
from nvdiffmodeling.src import mesh
from nvdiffmodeling.src import render
from nvdiffmodeling.src import util

import nvdiffrast.torch as dr

from utils.camera import get_camera_params
from utils.video import Video

import torch


#saves a rendered image (torch tensor) to an output path
#TODO: fix the saving the image so I don't have to make it a video ... 
def save_image (rendered_image, output_path) :
    img = rendered_image.clone()

    res = [512, 512]
    img = util.scale_img_nhwc(img, res)
    img = util.tonemap_srgb(img)
    
    img = img.detach().cpu().numpy()

    util.save_image(output_path, img)
    print("successfully saved image")

def main () :
    #==========================================
    #   Setting up parameters for rendering  
    #==========================================

    #variables needed for rendering:
    glctx = dr.RasterizeGLContext()
    device = torch.device("cuda:0")

    log_elev = 30.0
    rot_ang = 0
    log_dist = 10.0
    log_res = 512
    log_fov = 60

    log_light_power = 3.0

    params = get_camera_params(
        log_elev,
        rot_ang,
        log_dist,
        log_res,
        log_fov
    )

    #==========================================
    #   Preparaing mesh for rendering
    #==========================================


    load_mesh = obj.load_obj("objects/sphere.obj")

    # load_mesh = mesh.unit_size(load_mesh) #I think this normalizes it??

    load_mesh = mesh.compute_tangents(load_mesh)

    # aabb = mesh.aabb(load_mesh.eval())
    # scale = 2.0
    # final_mesh = mesh.center_by_reference(load_mesh.eval(params), aabb, scale) #i think this eval allows the mesh to have vertices again


    rendered_image = render.render_mesh(
        glctx,
        load_mesh.eval(params), #the eval allows the mesh to have v_pos again?
        params['mvp'],
        params['campos'],
        params['lightpos'],
        log_light_power,
        log_res
    )

    # save_image(rendered_image, "outputs/img.png")

    #this is fine for now 
    video = Video("output")
    video.ready_image(rendered_image)
    video.close()


if __name__== "__main__":
    main()
