use grid::Coordinate;
use proconio::{input, source::line::LineSource};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{stdin, BufReader, StdinLock};

use std::process::exit;

use grid::ADJACENTS;

use noise::Perlin;

fn main() {
    let stdin = stdin();
    let mut source = LineSource::new(BufReader::new(stdin.lock()));
    let input = Input::from_input(&mut source);
    solve(&mut source, input);
}

fn solve(stdin: &mut LineSource<BufReader<StdinLock>>, input: Input) {
    let mut dam = mat![0;input.n;input.n];
    let mut com = mat![false;input.n;input.n];
    let mut s_pred = mat![1000;input.n;input.n];
    let mut cost_sum = 0;
    for i in 0..input.h {
        let mut dist = mat![1 << 30;input.n;input.n];
        let mut from = mat![Coordinate(!0,!0);input.n;input.n];
        let mut que = BinaryHeap::new();
        dist[input.hs[i].0][input.hs[i].1] = 0;
        que.push((Reverse(0), input.hs[i]));
        while let Some((Reverse(d), p)) = que.pop() {
            if dist[p.0][p.1] < d {
                continue;
            }
            for &dxy in &ADJACENTS {
                let np = p + dxy;
                if np.in_map(input.n) {
                    // let cost = !com[np.0][np.1] as i32;
                    let cost = if com[np.0][np.1] {0}else {s_pred[np.0][np.1]};
                    if chmin!(dist[np.0][np.1], d + cost) {
                        que.push((Reverse(dist[np.0][np.1]), np));
                        from[np.0][np.1] = p;
                    }
                }
            }
        }
        let min_idx = input
            .ws
            .iter()
            .enumerate()
            .map(|(i, &x)| (dist[x.0][x.1], i))
            .min()
            .unwrap()
            .1;
        let mut cp = input.ws[min_idx];
        loop {
            if com[cp.0][cp.1] {
                if cp == input.hs[i] {
                    break;
                }
                cp = from[cp.0][cp.1];
                continue;
            }
            loop {
                let power = 100;
                let res = excavate(stdin, cp, power);
                dam[cp.0][cp.1] += power;
                cost_sum += power + input.c;
                if res == 1 {
                    com[cp.0][cp.1] = true;
                    break;
                } else if res == 2 {
                    eprintln!("Cost: {}", cost_sum);
                    exit(0);
                } else if res == -1 {
                    eprintln!("Wrong");
                    exit(1);
                }
            }
            com[cp.0][cp.1] = true;
            s_pred[cp.0][cp.1] = dam[cp.0][cp.1];
            for i in -20..=20 {
                for j in -20..=20 {
                    let nx = cp.0 as i32 + i;
                    let ny = cp.1 as i32 + j;
                    if nx < 0 || ny < 0 || nx >= input.n as i32 || ny >= input.n as i32 {
                        continue;
                    }
                    let np = Coordinate(nx as usize,ny as usize);
                    if com[np.0][np.1] {
                        continue;
                    }
                    let dist_cn = i.abs() + j.abs();
                    s_pred[np.0][np.1] = (s_pred[np.0][np.1] * dist_cn + s_pred[cp.0][cp.1]) / (dist_cn + 1);
                }
            }
            if cp == input.hs[i] {
                break;
            }
            cp = from[cp.0][cp.1];
        }
    }
    eprintln!("Finished");
}

fn excavate(stdin: &mut LineSource<BufReader<StdinLock>>, point: Coordinate, power: i32) -> i32 {
    println!("{} {} {}", point.0, point.1, power);
    input! {
        from stdin,
        res: i32
    };
    res
}

struct Input {
    n: usize,
    w: usize,
    h: usize,
    c: i32,
    ws: Vec<Coordinate>,
    hs: Vec<Coordinate>,
}
impl Input {
    fn from_input(stdin: &mut LineSource<BufReader<StdinLock>>) -> Self {
        input! {
            from stdin,
            n:usize,w:usize,h:usize,c:i32,ws:[(usize,usize);w],hs:[(usize,usize);h]
        };
        let ws = ws
            .iter()
            .map(|x| Coordinate::new(x.0, x.1))
            .collect::<Vec<_>>();
        let hs = hs
            .iter()
            .map(|x| Coordinate::new(x.0, x.1))
            .collect::<Vec<_>>();
        Self { n, w, h, c, ws, hs }
    }
}

pub mod grid {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Coordinate(pub usize, pub usize);
    impl Coordinate {
        pub const fn new(x: usize, y: usize) -> Self {
            Self(x, y)
        }
        pub fn in_map(&self, size: usize) -> bool {
            self.0 < size && self.1 < size
        }
        pub const fn to_index(&self, size: usize) -> usize {
            self.0 * size + self.1
        }
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CoordinateDiff(pub usize, pub usize);
    impl CoordinateDiff {
        pub const fn new(x: usize, y: usize) -> Self {
            Self(x, y)
        }
        pub const fn invert(&self) -> Self {
            Self::new(0usize.wrapping_sub(self.0), 0usize.wrapping_sub(self.1))
        }
    }
    impl std::ops::Add<CoordinateDiff> for Coordinate {
        type Output = Coordinate;
        fn add(self, rhs: CoordinateDiff) -> Self::Output {
            Self::Output::new(self.0.wrapping_add(rhs.0), self.1.wrapping_add(rhs.1))
        }
    }
    impl std::ops::AddAssign<CoordinateDiff> for Coordinate {
        fn add_assign(&mut self, rhs: CoordinateDiff) {
            self.0 = self.0.wrapping_add(rhs.0);
            self.1 = self.1.wrapping_add(rhs.1);
        }
    }
    pub const ADJACENTS: [CoordinateDiff; 4] = [
        CoordinateDiff::new(0, !0),
        CoordinateDiff::new(!0, 0),
        CoordinateDiff::new(0, 1),
        CoordinateDiff::new(1, 0),
    ];
}

#[macro_use]
mod macros {
    #[allow(unused_macros)]
    #[cfg(debug_assertions)]
    #[macro_export]
    macro_rules! debug {
        ( $x: expr, $($rest:expr),* ) => {
            eprint!(concat!(stringify!($x),": {:?}, "),&($x));
            debug!($($rest),*);
        };
        ( $x: expr ) => { eprintln!(concat!(stringify!($x),": {:?}"),&($x)); };
        () => { eprintln!(); };
    }
    #[allow(unused_macros)]
    #[cfg(not(debug_assertions))]
    #[macro_export]
    macro_rules! debug {
        ( $($x: expr),* ) => {};
        () => {};
    }
    #[macro_export]
    macro_rules! chmin {
        ($base:expr, $($cmps:expr),+ $(,)*) => {{
            let cmp_min = min!($($cmps),+);
            if $base > cmp_min {
                $base = cmp_min;
                true
            } else {
                false
            }
        }};
    }

    #[macro_export]
    macro_rules! chmax {
        ($base:expr, $($cmps:expr),+ $(,)*) => {{
            let cmp_max = max!($($cmps),+);
            if $base < cmp_max {
                $base = cmp_max;
                true
            } else {
                false
            }
        }};
    }

    #[macro_export]
    macro_rules! min {
        ($a:expr $(,)*) => {{
            $a
        }};
        ($a:expr, $b:expr $(,)*) => {{
            std::cmp::min($a, $b)
        }};
        ($a:expr, $($rest:expr),+ $(,)*) => {{
            std::cmp::min($a, min!($($rest),+))
        }};
    }
    #[macro_export]
    macro_rules! max {
        ($a:expr $(,)*) => {{
            $a
        }};
        ($a:expr, $b:expr $(,)*) => {{
            std::cmp::max($a, $b)
        }};
        ($a:expr, $($rest:expr),+ $(,)*) => {{
            std::cmp::max($a, max!($($rest),+))
        }};
    }

    #[macro_export]
    macro_rules! mat {
        ($e:expr; $d:expr) => { vec![$e; $d] };
        ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
    }
}
