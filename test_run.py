import hydra
from omegaconf import DictConfig, OmegaConf
from models import ETD_KT_CM_JAX_Vectorised

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_1 = OmegaConf.to_yaml(cfg)
    #print(cfg.model.c_0)# accessing c_0
    #print(cfg.model['c_1'])
    #params = ConfigDict(KDV_params)
    params = cfg.model
    print(params)
    fwd_model = ETD_KT_CM_JAX_Vectorised(params)

    
if __name__ == "__main__":
    main()