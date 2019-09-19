use hurban_selector::geometry;
use hurban_selector::operations::shrink_wrap::{ShrinkWrapOp, ShrinkWrapParams};
use hurban_selector::renderer::SceneRendererGeometry;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_shrink_wrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrink_wrap");
    let geometry = geometry::uv_sphere([0.0, 0.0, 0.0], 1.0, 3, 3);

    for density in (5u32..=25u32).step_by(5) {
        group.bench_with_input(
            BenchmarkId::new("Batch + Convert", density),
            &density,
            |b, density| {
                b.iter(|| {
                    let mut op = ShrinkWrapOp::new(ShrinkWrapParams {
                        geometry: geometry.clone(),
                        sphere_density: black_box(*density),
                        step: 0,
                    });

                    let mut res = None;
                    while let Some(value) = op.next_value() {
                        res = Some(SceneRendererGeometry::from_geometry(&value));
                    }

                    res
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Iterative(5) + Convert", density),
            &density,
            |b, density| {
                b.iter(|| {
                    let mut op = ShrinkWrapOp::new(ShrinkWrapParams {
                        geometry: geometry.clone(),
                        sphere_density: black_box(*density),
                        step: 5,
                    });

                    let mut res = None;
                    while let Some(value) = op.next_value() {
                        res = Some(SceneRendererGeometry::from_geometry(&value));
                    }

                    res
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Batch", density),
            &density,
            |b, density| {
                b.iter(|| {
                    let mut op = ShrinkWrapOp::new(ShrinkWrapParams {
                        geometry: geometry.clone(),
                        sphere_density: black_box(*density),
                        step: 0,
                    });

                    let mut res = None;
                    while let Some(value) = op.next_value() {
                        res = Some(value);
                    }

                    res
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Iterative(5)", density),
            &density,
            |b, density| {
                b.iter(|| {
                    let mut op = ShrinkWrapOp::new(ShrinkWrapParams {
                        geometry: geometry.clone(),
                        sphere_density: black_box(*density),
                        step: 5,
                    });

                    let mut res = None;
                    while let Some(value) = op.next_value() {
                        res = Some(value);
                    }

                    res
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_shrink_wrap);
criterion_main!(benches);
