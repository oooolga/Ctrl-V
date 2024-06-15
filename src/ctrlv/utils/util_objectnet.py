import torch

def convertObjects(obj):

    # obj[key] shape: (Batch, clip_length, 77, value_dim)
    #
    # Values   Name         Description
    # ----------------------------------------------------------------------------
    #     1    type         Describes the type of object: 'Car', 'Van', 'Truck',
    #                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    #                         'Misc' or 'DontCare'
    #     1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
    #                         truncated refers to the object leaving image boundaries
    #     1    occluded     Integer (0,1,2,3) indicating occlusion state:
    #                         0 = fully visible, 1 = partly occluded
    #                         2 = largely occluded, 3 = unknown
    #     1    alpha        Observation angle of object, ranging [-pi..pi]
    #     4    bbox         2D bounding box of object in the image (0-based index):
    #                         contains left, top, right, bottom pixel coordinates
    #     3    dimensions   3D object dimensions: height, width, length (in meters)
    #     3    locations     3D object location x,y,z in camera coordinates (in meters)
    #     1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    #     1    score        Only for results: Float, indicating confidence in
    #                         detection, needed for p/r curves, higher is better.
    #     0    track_id     no dim, ids are written in the 3rd dim of length 77
    CLASS_IDS_LOOKUP = {'Car':1,
                        'Van':2,
                        'Truck':3,
                        'Pedestrian':4,
                        'Person':5,
                        'Cyclist':6,
                        'Tram':7,
                        'Misc':8,
                        'DontCare':9
                        }
    # type = [[CLASS_IDS_LOOKUP[l] for l in ll if l is not None else 0] for ll in obj['type']]
    return torch.cat([obj['track_id'].view(*obj['track_id'].shape,1),
               obj['truncated'].view(*obj['truncated'].shape,1),
               obj['occluded'].view(*obj['occluded'].shape,1),
               obj['alpha'].view(*obj['alpha'].shape,1),
               obj['bbox'],
               obj['dimensions'],
               obj['locations'],
               obj['rotation_y'].view(*obj['rotation_y'].shape,1),], -1)

def revertEmbed(embed):
    # return torch.cat([obj['track_id'].view(*obj['track_id'].shape,1),
    #            obj['truncated'].view(*obj['truncated'].shape,1),
    #            obj['occluded'].view(*obj['occluded'].shape,1),
    #            obj['alpha'].view(*obj['alpha'].shape,1),
    #            obj['bbox'],
    #            obj['dimensions'],
    #            obj['locations'],
    #            obj['rotation_y'].view(*obj['rotation_y'].shape,1),], -1)
    obj = {'track_id':   embed[:,:,:,0],
           'truncated':  embed[:,:,:,1],
           'occluded':   embed[:,:,:,2],
           'alpha':      embed[:,:,:,3],
           'bbox':       embed[:,:,:,4:8],
           'dimensions': embed[:,:,:,8:11],
           'locations':  embed[:,:,:,11:14],
           'rotation_y':  embed[:,:,:,14],
           }
    return obj

def generate_step(model, all_embed, max_len):
    input_embed = all_embed[:,:1]
    shape = input_embed.shape
    cur_len = shape[1]
    for idx in range(cur_len,max_len):
        output = model(inputs_embeds=input_embed.view(*(input_embed.shape[:2]),-1),
        labels=None,return_dict=False)
        pred = output[1][:,-1:,:].detach()
        input_embed = torch.cat([input_embed,pred.view(shape[0],1,shape[2],shape[3])],1)
        print(idx)
    return revertEmbed(input_embed.cpu())