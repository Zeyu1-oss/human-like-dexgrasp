import numpy as np

from .base import build_grasp_matrix


def calcu_dfc_metric(contact_pos, contact_normal, miu_coef, enable_density=False):
    grasp_matrix = build_grasp_matrix(contact_pos, contact_normal)
    if enable_density:
        cos_theta = (contact_normal[:, None, :] * contact_normal[:, :, None]).sum(
            axis=-1
        )
        density = (
            1
            / np.clip(
                np.clip(cos_theta, a_min=0, a_max=100).sum(axis=-1),
                a_min=1e-4,
                a_max=100,
            )[:, None]
         # 取抓取矩阵的前3行（力分量部分），按密度加权后求范数
        )
        dfc_metric = np.linalg.norm(np.sum(grasp_matrix[:, :, 0] * density, axis=0))
    else:
        # 取抓取矩阵的前3行（力分量部分），直接求和后求范数
        dfc_metric = np.linalg.norm(np.sum(grasp_matrix[:, :, 0], axis=0))
    return dfc_metric
