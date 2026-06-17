from .heston import simulate as heston_sim
# from .quintic_1f import simulate as quintic_1f_sim
# from .rough_bergomi import simulate as rbergomi_sim
# ... import others as you build them

# Create a master dictionary
MODEL_REGISTRY = {
    "Heston": heston_sim,
    # "1F_Quintic_OU": quintic_1f_sim,
    # "Rough_Bergomi": rbergomi_sim,
}