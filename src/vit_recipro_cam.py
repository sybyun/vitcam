import copy
import torch


class ViTReciproCam:
    '''
    ViTReciproCam class contains official implementation of Reciprocal CAM algorithm for ViT architecture 
    published at https://arxiv.org/pdf/2310.02588.
    '''

    def __init__(self, model, device, target_layer_name=None, is_gaussian=True, block_index=-2, cls_token=True):
        '''
        Creator of ViTReciproCAM class
        
        Args:
            model: CNN architectur pytorch model
            device: runtime device type (ex, 'cuda', 'cpu')
            target_layer_name: layer name for understanding the layer's activation
            is_gaussian: boolean value for using gaussian filter
            block_index: encoding block index for using CAM
            cls_token: boolean value for using class token
        '''

        self.model = copy.deepcopy(model)
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.device = device
        self.is_gaussian = is_gaussian
        self.cls_token = cls_token
        self.softmax = torch.nn.Softmax(dim=1)
        self.feature = None
        self.target_layers = []
        self._find_target_layer()
        if block_index == -1:
            print('Last block layer cannot be used for CAM, so using the second last block layer.')
            block_index = -2
        self.target_layers[block_index].register_forward_hook(self._cam_hook())
                
        filter = [[1/16.0, 1/8.0, 1/16.0],
                    [1/8.0, 1/4.0, 1/8.0],
                    [1/16.0, 1/8.0, 1/16.0]]
        self.gaussian = torch.tensor(filter).to(device)
        
        
    def _find_target_layer(self):
        '''
        Searching target layer by name from given network model.
        '''
        if self.target_layer_name:
            for name, module in self.model.named_modules():
                if self.target_layer_name in name:
                    self.target_layers.append(module)        
        else:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.LayerNorm):
                    self.target_layer_name = name.split('.')[-1]
                    break
            for name, module in self.model.named_modules():
                if self.target_layer_name in name:
                    self.target_layers.append(module)        
                    
                    
    def _cam_hook(self):
        '''
        Setup hook funtion for generating new masked features for calculating reciprocal activation score 
        '''

        def fn(_, input, output):
            self.feature = output[0].unsqueeze(0)
            bs, tn, dim = self.feature.shape
            new_features = self._generate_masked_feature(tn, dim)
            new_features = torch.cat((self.feature, new_features), dim = 0)
            return new_features

        return fn


    def _generate_masked_feature(self, tn, dim):
        '''
        Generate spatially masked feature map [h*w, nc, h, w] from input feature map [1, nc, h, w].
        If is_gaussian is true then the spatially masked feature map's value are filtered by 3x3 Gaussian filter.  
        '''
        
        new_outputs = torch.zeros(tn-1, tn, dim).to(self.device)
        if self.is_gaussian == False:
            for i in range(tn-1):
                if self.cls_token == True:
                    new_outputs[i, 0, :] = self.feature[0, 0, :]        
                new_outputs[i, i+1, :] = self.feature[0, i+1, :]
        else:
            n_c = int((tn-1)**0.5)
            spatial_feature = self.feature[0,1:,:]
            spatial_feature = spatial_feature.reshape(1, n_c, n_c, dim)
            if self.cls_token == True:
                new_outputs[:,0,:] = self.feature[0,0,:]
            new_outputs_r = new_outputs[:,1:,:]
            new_outputs_r = new_outputs_r.reshape(tn-1,n_c,n_c,dim)
            score_map = self.gaussian.reshape(3,3,1).repeat(1,1,dim)
            for i in range(n_c):
                ky_s = max(i-1, 0)
                ky_e = min(i+1, n_c-1)
                if i == 0: sy_s = 1
                else: sy_s = 0
                if i == n_c-1: sy_e = 1
                else: sy_e = 2
                for j in range(n_c):
                    kx_s = max(j-1, 0)
                    kx_e = min(j+1, n_c-1)
                    if j == 0: sx_s = 1
                    else: sx_s = 0
                    if j == n_c-1: sx_e = 1
                    else: sx_e = 2
                    new_outputs_r[i*n_c+j, ky_s:ky_e+1, kx_s:kx_e+1, :] \
                        = spatial_feature[0, ky_s:ky_e+1, kx_s:kx_e+1, :] \
                        * score_map[sy_s:sy_e+1, sx_s:sx_e+1, :]
            new_outputs_r = new_outputs_r.reshape(tn-1,tn-1,dim)
        return new_outputs
            

    def _get_token_weight(self, predictions, index, num_tokens):
        '''
        Calculate class activation map from the prediction result of mosaic feature input.
        '''
                
        n_c = int((num_tokens-1)**0.5)
        weight = (predictions[:, index]).reshape((n_c, n_c))   
        weight_min = weight.min()
        diff = weight.max() - weight_min
        weight = (weight - weight_min) / (diff)
 
        return weight


    def __call__(self, input_tensor, index=None):    

        with torch.no_grad():
            predictions = self.model(input_tensor)
            predictions = self.softmax(predictions)

            if index == None:
                index = predictions[0].argmax().item()

            bs, t_n, dim = self.feature.shape
            cam = self._get_token_weight(predictions[1:, :], index, t_n)
            cam = cam.detach()

        return cam, index
