ns-train instant-ngp \
--vis wandb --viewer.skip-openrelay True \
--method-name instant-ngp_far5_lr1e-2-3 \
--pipeline.model.far-plane 5 \
nerfstudio-data --data data/nerfstudio/poster \


ns-train instant-ngp \                                                                                                                               
--vis wandb --viewer.skip-openrelay True \                                                                                                             
--method-name instant-ngp_far20_lr1e-2-3 \                                                                                                             
--pipeline.model.far-plane 20 \                                                                                                                        
nerfstudio-data --data data/nerfstudio/bww_entrance \                                                                                                 