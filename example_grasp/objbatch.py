import bpy
import os
import glob

# 设置你的根目录
root_dir = "/media/rose/KINGSTON/obj/vis_obj"

# 清空当前场景
bpy.ops.wm.read_factory_settings(use_empty=True)

offset_x = 0.0
step = 3.0  # 每组偏移量

for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    # 遍历 scale*/pose* 子目录
    for scale_pose_dir in sorted(glob.glob(os.path.join(folder_path, "*", "*"))):
        if not os.path.isdir(scale_pose_dir):
            continue

        # 查找第一个 obj 和 grasp 文件
        obj_files = sorted(glob.glob(os.path.join(scale_pose_dir, "*_obj.obj")))
        grasp_files = sorted(glob.glob(os.path.join(scale_pose_dir, "*_grasp_*.obj")))

        if not obj_files or not grasp_files:
            continue

        # 只导入第一个 obj 文件
        bpy.ops.import_scene.obj(filepath=obj_files[0])
        for obj in bpy.context.selected_objects:
            obj.location[0] += offset_x

        # 只导入第一个 grasp 文件
        bpy.ops.import_scene.obj(filepath=grasp_files[0])
        for obj in bpy.context.selected_objects:
            obj.location[0] += offset_x

        offset_x += step  # 下一组右移

print("✅ 已加载所有物体和抓取")
