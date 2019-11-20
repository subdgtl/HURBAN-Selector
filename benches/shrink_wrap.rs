use hurban_selector::geometry;
use hurban_selector::operations::shrink_wrap::{self, ShrinkWrapParams};
use hurban_selector::renderer::SceneRendererGeometry;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_shrink_wrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrink_wrap");
    let geometry = geometry::uv_sphere_geometry([0.0, 0.0, 0.0], 1.0, 3, 3);

    for density in (10u32..=50u32).step_by(10) {
        group.bench_with_input(
            BenchmarkId::new("shrink_wrap (batch)", density),
            &density,
            |b, density| {
                b.iter(|| {
                    shrink_wrap::shrink_wrap(ShrinkWrapParams {
                        geometry: &geometry,
                        sphere_density: black_box(*density),
                    })
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("shrink_wrap (batch + convert)", density),
            &density,
            |b, density| {
                b.iter(|| {
                    let value = shrink_wrap::shrink_wrap(ShrinkWrapParams {
                        geometry: &geometry,
                        sphere_density: black_box(*density),
                    });

                    SceneRendererGeometry::from_geometry(&value)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_shrink_wrap);
criterion_main!(benches);
