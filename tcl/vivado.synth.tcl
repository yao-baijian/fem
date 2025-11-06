# 设置项目参数
# set rtl_file /home/byao/Desktop/Benchmarks/ISCAS85/c1355/c1355.v
# set top_module {c1355}
# set part_name {xcvu065-ffvc1517-1-i}
# set output_dir {output_dir}

# set rtl_file /home/byao/Desktop/Benchmarks/ISCAS85/c1908/c1908.v
# set top_module {c1908}
# set part_name {xcvu065-ffvc1517-1-i}
# set output_dir {output_dir}

set rtl_file /home/byao/Desktop/Benchmarks/ISCAS89/s9234.v
set top_module {s9234}
set part_name {xcvu065-ffvc1517-1-i}
set output_dir {output_dir}


# 创建临时项目
create_project -part $part_name -force temp_project ./temp_project

# 添加源文件
add_files -norecurse $rtl_file

# 设置顶层模块
set_property top $top_module [current_fileset]

# 运行综合
synth_design -top $top_module -part $part_name -flatten_hierarchy rebuilt

# 生成综合后的设计检查点
write_checkpoint -force $output_dir/post_synth.dcp

opt_design

place_design

route_design

write_checkpoint -force $output_dir/post_impl.dcp

# 关闭项目
close_project