pub static DIFF_SPACE_SRC: &'static str = r#"
typedef struct {
    long2 input_ones;
    long2 input_zeros;
} Cluster;

typedef struct {
    Cluster clusters[10];
} Point;

kernel void add_idx(global Point* in_points)
{
    uint const idx = get_global_id(0);
    global Point* const point = &in_points[idx];
    for (uint cid = 0; cid < 10; ++cid) {
        point->clusters[cid].input_ones.x += idx + cid;
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct Point {
    clusters: [Cluster; 10],
}
unsafe impl ocl::core::OclPrm for Point {}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct Cluster {
    input_ones: ocl::prm::Ulong2,
    input_zeros: ocl::prm::Ulong2,
}
unsafe impl ocl::core::OclPrm for Cluster {}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("{:?}", args);

    let diff_space_len: usize = args[1].parse().unwrap();
    let pro_que = ocl::ProQue::builder()
        .src(DIFF_SPACE_SRC)
        .dims(diff_space_len)
        .build().unwrap();
    dbg!(std::mem::size_of::<Point>());
    dbg!(std::mem::align_of::<Point>());

    let mut vec = vec![Point::default(); diff_space_len];
    let buffer = pro_que.buffer_builder::<Point>()
        .len(diff_space_len)
        .copy_host_slice(&vec)
        .build()
        .unwrap();

    let now = std::time::Instant::now();
    let kernel = pro_que.kernel_builder("add_idx")
        .arg(&buffer)
        .build().unwrap();

    unsafe { kernel.enq().unwrap(); }
    dbg!(now.elapsed());

    buffer.read(&mut vec).enq().unwrap();

    println!("The value at index [{}] is now '{:?}'!", 3, vec[3]);
    println!("The value at index [{}] is now '{:?}'!", diff_space_len-3, vec[diff_space_len-3]);
}
