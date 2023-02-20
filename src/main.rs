use grid::Coordinate;
use proconio::{input, source::line::LineSource};
use std::cmp::Reverse;
use std::collections::{BinaryHeap,VecDeque};
use std::io::{stdin, BufReader, StdinLock};

use std::process::exit;

use grid::ADJACENTS;

fn main() {
    let stdin = stdin();
    let mut source = LineSource::new(BufReader::new(stdin.lock()));
    let input = Input::from_input(&mut source);
    solve(&mut source, input);
}

fn solve(stdin: &mut LineSource<BufReader<StdinLock>>, input: Input) {
    let mut dam = mat![0;input.n;input.n];
    let mut com = mat![false;input.n;input.n];
    let mut cost_sum = 0;
    let mut to_w = mat![1 << 30;input.n;input.n];
    let mut que = VecDeque::new();
    macro_rules! excavate_proc {
        ($point: expr,$power: expr) => {{
            let res = excavate(stdin,$point,$power);
            cost_sum += $power;
            dam[$point.0][$point.1] += $power;
            if res == 1 {
                com[$point.0][$point.1] = true;
            }
            else if res == 2 {
                eprintln!("Cost sum: {}",cost_sum);
                exit(0);
            }
            res
        }};
    }
    for i in 0..input.w {
        to_w[input.ws[i].0][input.ws[i].1] = 0;
        que.push_back(input.ws[i]);
    }
    while let Some(p) = que.pop_front() {
        for &cd in &ADJACENTS {
            let np = p + cd;
            if np.in_map(input.n) && chmin!(to_w[np.0][np.1],to_w[p.0][p.1] + 1) {
                que.push_back(np);
            }
        }
    }

    let log2c = match input.c {
        1=>0,2=>1,4=>2,8=>3,16=>4,32=>5,64=>6,128=>7,_=>unreachable!()
    };
    let mut que = BinaryHeap::new();
    // 家の岩盤を掘る
    for i in 0..input.h {
        let power = 20 + log2c;
        loop {
            let res = excavate_proc!(input.hs[i],power);
            if res == 1 {
                que.push((Reverse(to_w[input.hs[i].0][input.hs[i].1]),input.hs[i]));
                break;
            }
        }
    }
    let mut done = vec![false;input.h];
    // 水場に近い家から水場に向かって伸ばす
    for _ in 0..input.h {
        let (idx,mut cur) = {
            let mut mn = 1 << 30;
            let mut idx = !0;
            for i in 0..input.h {
                if !done[i] && chmin!(mn,to_w[input.hs[i].0][input.hs[i].1]) {
                    idx = i;
                }
            }
            (idx,input.hs[idx])
        };
        let mut vis = mat![false;input.n;input.n];
        let mut moved = BinaryHeap::new();
        let mut cnt = 0;        
        vis[cur.0][cur.1] = true;
        // 水場に付くまで移動を繰り返す
        // 移動は出来る限り水場に近づいていくように2,3方向への移動に限定する
        // 許容の差が大きくなっていく
        // 一定時間移動、移動できなくなったら水場に一番近い場所を選び直して再スタート
        'outer: loop {
            let mut ord = vec![];
            let cur_dam = dam[cur.0][cur.1];
            cnt += 1;
            for &cd in &ADJACENTS {
                let np = cur + cd;
                if np.in_map(input.n) {
                    if vis[np.0][np.1] || to_w[cur.0][cur.1] < to_w[np.0][np.1] {
                        continue;
                    }
                    ord.push((to_w[np.0][np.1],np));
                }
            }
            if ord.is_empty() {
                // cnt += 1;
                for &cd in &ADJACENTS {
                    let np = cur + cd;
                    if np.in_map(input.n) && to_w[np.0][np.1] < to_w[cur.0][cur.1] {
                        cur = np;
                        break;
                    }
                }
                continue;
            }
            ord.sort();
            if ord[0].0 == 0 {
                ord = vec![ord[0]];
            }
            for &(w_dist,np) in &ord {
                // 水源だったらそこに移動しよう
                if w_dist == 0 {
                    if com[np.0][np.1] {
                        break 'outer;
                    }
                    let d = dam[np.0][np.1];
                    let attempt = (if cur_dam < 100 {(cur_dam - 10).max(20) - d} else{cur_dam - 100 - d}).min(5000);
                    let res = excavate_proc!(np,attempt);
                    if res == 1 {
                        break 'outer;
                    }
                    // いけるまでやる
                    let attempt = if cur_dam < 100 {20}else{100};
                    loop {
                        let res = excavate_proc!(np,attempt);
                        if res == 1 {
                            break 'outer;
                        }
                    }
                }
                // 差の許容量
                let allow = (50 + cnt - if w_dist - 1 == to_w[cur.0][cur.1] {0}else{50}).max(0);
                let d = dam[np.0][np.1];
                // すでにたくさん削っていたら見送る
                if allow + cur_dam <= d {
                    continue;
                }
                let attempt = (if cur_dam < 100 {(cur_dam - 10).max(20) - d} else{cur_dam - 100 - d}).min(5000);
                // とりあえず少し掘ってみる
                if attempt > 20 {
                    let res = excavate_proc!(np,attempt);
                    if res == 1 {
                        moved.push(np);
                        // cnt += 1;
                        cur = np;
                        vis[np.0][np.1] = true;
                        continue 'outer;
                    }
                }
                let d = dam[np.0][np.1];
                let attempt = (allow + cur_dam - d).max(20).min(5000);
                let res = excavate_proc!(np,attempt);
                if res == 1 {
                    moved.push(np);
                    // cnt += 1;
                    cur = np;
                    vis[np.0][np.1] = true;
                    continue 'outer;
                }
            }
        }
        // 水場の更新
        to_w = mat![1 << 30;input.n;input.n];
        let mut que = VecDeque::new();
        for i in 0..input.w {
            to_w[input.ws[i].0][input.ws[i].1] = 0;
            que.push_back(input.ws[i]);
        }
        while let Some(p) = que.pop_front() {
            for &cd in &ADJACENTS {
                let np = p + cd;
                if np.in_map(input.n) && com[np.0][np.1] && chmin!(to_w[np.0][np.1],to_w[p.0][p.1]) {
                    que.push_back(np);
                }
            }
        }
        for i in 0..input.n {
            for j in 0..input.n {
                if to_w[i][j] == 0 {
                    que.push_back(Coordinate(i, j));
                }
            }
        }
        while let Some(p) = que.pop_front() {
            for &cd in &ADJACENTS {
                let np = p + cd;
                if np.in_map(input.n) && chmin!(to_w[np.0][np.1],to_w[p.0][p.1] + 1) {
                    que.push_back(np);
                }
            }
        }
        done[idx] = true;
    }
}

fn excavate(stdin: &mut LineSource<BufReader<StdinLock>>, point: Coordinate, power: i32) -> i32 {
    println!("{} {} {}", point.0, point.1, power);
    input! {
        from stdin,
        res: i32
    };
    if res == -1 {
        eprintln!("Wrong");
        exit(1);
    }
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
