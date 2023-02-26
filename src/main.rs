use grid::Coordinate;
use proconio::{input, source::line::LineSource};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::io::{stdin, BufReader, StdinLock};

use std::process::exit;
use std::env;

use grid::{ADJACENTS, ADJACENTS8};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let hp = HyperParameter::from_args(args);
    let stdin = stdin();
    let mut source = LineSource::new(BufReader::new(stdin.lock()));
    let input = Input::from_input(&mut source);
    solve(&mut source, input, hp);
}

// 最初に全体を掘削するのでは無く、家からのパスを見つけるために毎回掘削の範囲を広げ、強度を上げていく
fn solve(stdin: &mut LineSource<BufReader<StdinLock>>, input: Input, hp: HyperParameter) {
    // 全体を何分割かしてそこの地層の強度を調べる
    let from_idx = |x: usize| if x == hp.SEPN - 1 { input.n - 1 } else { x * hp.SEP };
    let mut dam: Vec<Vec<i32>> = mat![0;input.n;input.n];
    let mut com = mat![false;input.n;input.n];
    let mut cost_sum = 0;
    macro_rules! excavate_proc {
        ($point: expr,$power: expr) => {{
            let res = excavate(stdin, $point, $power);
            cost_sum += $power;
            dam[$point.0][$point.1] += $power;
            if res == 1 {
                com[$point.0][$point.1] = true;
            } else if res == 2 {
                eprintln!("Cost sum: {}", cost_sum);
                exit(0);
            }
            res
        }};
    }
    let mut uf = acl::Dsu::new(hp.SEPN * hp.SEPN + 1 + input.h);

    #[allow(non_snake_case)]
    let WNODE: usize = hp.SEPN * hp.SEPN;
    #[allow(non_snake_case)]
    let HNODE_OFFSET: usize = hp.SEPN * hp.SEPN + 1;

    for i in 0..input.w {
        let hx = input.ws[i].0 / hp.SEP;
        let hy = input.ws[i].1 / hp.SEP;
        uf.merge(WNODE, hx * hp.SEPN + hy);
        uf.merge(WNODE, (hx + 1) * hp.SEPN + hy);
        uf.merge(WNODE, hx * hp.SEPN + (hy + 1));
        uf.merge(WNODE, (hx + 1) * hp.SEPN + (hy + 1));
    }
    for i in 0..input.h {
        let hx = input.hs[i].0 / hp.SEP;
        let hy = input.hs[i].1 / hp.SEP;
        uf.merge(HNODE_OFFSET + i, hx * hp.SEPN + hy);
        uf.merge(HNODE_OFFSET + i, (hx + 1) * hp.SEPN + hy);
        uf.merge(HNODE_OFFSET + i, hx * hp.SEPN + (hy + 1));
        uf.merge(HNODE_OFFSET + i, (hx + 1) * hp.SEPN + (hy + 1));
    }
    // まず最初は300で掘削、その後範囲を一つ広げ、300で掘削し、そのほかを300さらに掘り進める
    // 距離はマンハッタン
    for i in 0..input.h {
        let ox = input.hs[i].0 / hp.SEP;
        let oy = input.hs[i].1 / hp.SEP;
        let mut lx = ox;
        let mut ly = oy;
        let mut rx = ox;
        let mut ry = oy;
        let mut s = 0;
        while !uf.same(WNODE, HNODE_OFFSET + i) {
            s += 1;
            for x in lx..=rx {
                for y in ly..=ry {
                    let hxi = from_idx(x);
                    let hyi = from_idx(y);
                    if com[hxi][hyi] {
                        continue;
                    }
                    let d = abs_diff(x, ox).max(abs_diff(y, oy));
                    let tar = (s - d) as i32 * hp.DIFF1;
                    while dam[hxi][hyi] < tar {
                        let res = excavate_proc!(Coordinate(hxi, hyi), hp.EX_STEP);
                        if res == 1 {
                            break;
                        }
                    }
                    if !com[hxi][hyi] {
                        continue;
                    }
                    for &cd in &ADJACENTS8 {
                        let nx = Coordinate(x, y) + cd;
                        if nx.in_map(hp.SEPN) && com[from_idx(nx.0)][from_idx(nx.1)] {
                            uf.merge(x * hp.SEPN + y, nx.0 * hp.SEPN + nx.1);
                        }
                    }
                }
            }
            lx = if lx == 0 { 0 } else { lx - 1 };
            ly = if ly == 0 { 0 } else { ly - 1 };
            rx = if rx == hp.SEPN - 1 { hp.SEPN - 1 } else { rx + 1 };
            ry = if ry == hp.SEPN - 1 { hp.SEPN - 1 } else { ry + 1 };
        }
    }
    eprintln!("Mining part: {}", cost_sum);

    let mut to_w = mat![1 << 30;input.n;input.n];
    let mut que = VecDeque::new();
    for i in 0..input.w {
        to_w[input.ws[i].0][input.ws[i].1] = 0;
        if com[input.ws[i].0][input.ws[i].1] {
            que.push_back(input.ws[i]);
        }
    }
    while let Some(p) = que.pop_front() {
        for &cd in &ADJACENTS {
            let np = p + cd;
            if np.in_map(input.n) && chmin!(to_w[np.0][np.1], to_w[p.0][p.1] + 1) {
                que.push_back(np);
            }
        }
    }
    let mut s_pred = mat![!0;input.n;input.n];
    // 近い四隅から値を持ってきて予測
    for i in 0..input.n {
        for j in 0..input.n {
            let hx = i / hp.SEP;
            let hy = j / hp.SEP;
            let hxi = from_idx(hx);
            let hyi = from_idx(hy);
            let hx1i = from_idx(hx + 1);
            let hy1i = from_idx(hy + 1);
            let hxhy = if com[hxi][hyi] {
                dam[hxi][hyi]
            } else {
                hp.UNKNOWN
            };
            let hxhy1 = if com[hxi][hy1i] {
                dam[hxi][hy1i]
            } else {
                hp.UNKNOWN
            };
            let hx1hy = if com[hx1i][hyi] {
                dam[hx1i][hyi]
            } else {
                hp.UNKNOWN
            };
            let hx1hy1 = if com[hx1i][hy1i] {
                dam[hx1i][hy1i]
            } else {
                hp.UNKNOWN
            };
            let d0 = (hxhy * (hy1i - j) as i32 + hxhy1 * (j - hyi) as i32) / hp.SEP as i32;
            let d1 = (hx1hy * (hy1i - j) as i32 + hx1hy1 * (j - hyi) as i32) / hp.SEP as i32;
            let d2 = (d0 * (hx1i - i) as i32 + d1 * (i - hxi) as i32) / hp.SEP as i32;
            s_pred[i][j] = d2;
        }
    }
    visualize(&input, &s_pred);

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
            for &cd in &ADJACENTS {
                let np = p + cd;
                if np.in_map(input.n) {
                    let nx =
                        d + if com[np.0][np.1] {
                            0
                        } else {
                            s_pred[np.0][np.1]
                        } + input.c;
                    if chmin!(dist[np.0][np.1], nx) {
                        que.push((Reverse(dist[np.0][np.1]), np));
                        from[np.0][np.1] = p;
                    }
                }
            }
        }
        let mut tar = Coordinate(!0, !0);
        // 予測距離 グリッドでの距離
        let mut now = (1 << 30, 1 << 30);
        for i in 0..input.n {
            for j in 0..input.n {
                if to_w[i][j] != 0 {
                    continue;
                }
                if chmin!(now, (dist[i][j], to_w[i][j])) {
                    tar = Coordinate(i, j);
                }
            }
        }
        let mut cur = tar;
        let mut path = vec![];
        // たどり着くまで
        loop {
            path.push(cur);
            if cur == input.hs[i] {
                break;
            }
            cur = from[cur.0][cur.1];
        }
        path.reverse();
        for p in path {
            if com[p.0][p.1] {
                continue;
            }
            // その場を掘る
            loop {
                let rem = s_pred[p.0][p.1] - dam[p.0][p.1];
                let attempt = if rem < hp.PRED_BORDER { hp.LOW_DAMAGE } else { hp.HIGH_DAMAGE };
                let res = excavate_proc!(p, attempt);
                if res == 1 {
                    break;
                }
            }
        }
        // 水場の更新
        to_w = mat![1 << 30;input.n;input.n];
        let mut que = VecDeque::new();
        for i in 0..input.w {
            to_w[input.ws[i].0][input.ws[i].1] = 0;
            if com[input.ws[i].0][input.ws[i].1] {
                que.push_back(input.ws[i]);
            }
        }
        while let Some(p) = que.pop_front() {
            for &cd in &ADJACENTS {
                let np = p + cd;
                if np.in_map(input.n) && com[np.0][np.1] && chmin!(to_w[np.0][np.1], to_w[p.0][p.1])
                {
                    que.push_back(np);
                }
            }
        }
        for i in 0..input.n {
            for j in 0..input.n {
                if to_w[i][j] == 0 && com[i][j] {
                    que.push_back(Coordinate(i, j));
                }
            }
        }
        while let Some(p) = que.pop_front() {
            for &cd in &ADJACENTS {
                let np = p + cd;
                if np.in_map(input.n) && chmin!(to_w[np.0][np.1], to_w[p.0][p.1] + 1) {
                    que.push_back(np);
                }
            }
        }
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

#[cfg(feature = "visualize")]
fn visualize(input: &Input, s_pred: &Vec<Vec<i32>>) {
    eprintln!("start visualizing");
    for i in 0..input.n {
        for j in 0..input.n {
            eprint!("{}", s_pred[i][j]);
            eprint!("{}", if j == input.n - 1 { "\n" } else { " " });
        }
    }
    eprintln!("end visualizing");
}
#[cfg(not(feature = "visualize"))]
fn visualize(_input: &Input, _s_pred: &Vec<Vec<i32>>) {}

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

#[allow(non_snake_case)]
struct HyperParameter {
    // SEPコのブロックで分割
    SEP: usize,
    SEPN: usize,
    UNKNOWN: i32,
    // 距離が1伸びた時の差
    DIFF1: i32,
    // 毎回の掘削でどれだけ掘るか
    EX_STEP: i32,
    // 予測値のボーダー
    PRED_BORDER: i32,
    // それ未満だったら
    LOW_DAMAGE: i32,
    // それ以上だったら
    HIGH_DAMAGE: i32,
}

impl HyperParameter {
    fn from_args(args: Vec<String>) -> Self {
        let l = args.len();
        if l == 0 {
            Self {
                SEP: 20,
                SEPN: 11,
                UNKNOWN: 5507,
                DIFF1: 119,
                EX_STEP: 78,
                PRED_BORDER: 631,
                LOW_DAMAGE: 57,
                HIGH_DAMAGE: 155,
            }
        }
        else {
            assert_eq!(l,7);
            let sep = args[0].parse::<usize>().unwrap();
            let sepn = (200 + sep - 1) / sep + 1;
            let unknown = args[1].parse::<i32>().unwrap();
            let diff1 = args[2].parse::<i32>().unwrap();
            let ex_step = args[3].parse::<i32>().unwrap();
            let pred_border = args[4].parse::<i32>().unwrap();;
            let low_damage = args[5].parse::<i32>().unwrap();;
            let high_damage = args[6].parse::<i32>().unwrap();;
            Self {
                SEP: sep,
                SEPN: sepn,
                UNKNOWN: unknown,
                DIFF1: diff1,
                EX_STEP: ex_step,
                PRED_BORDER: pred_border,
                LOW_DAMAGE: low_damage,
                HIGH_DAMAGE: high_damage
            }
        }
    }
}

fn abs_diff(a: usize, b: usize) -> usize {
    if a < b {
        b - a
    } else {
        a - b
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
    pub const ADJACENTS8: [CoordinateDiff; 8] = [
        CoordinateDiff::new(0, !0),
        CoordinateDiff::new(!0, 0),
        CoordinateDiff::new(0, 1),
        CoordinateDiff::new(1, 0),
        CoordinateDiff::new(!0, !0),
        CoordinateDiff::new(!0, 1),
        CoordinateDiff::new(1, !0),
        CoordinateDiff::new(1, 1),
    ];
}

pub mod acl {
    pub struct Dsu {
        n: usize,
        // root node: -1 * component size
        // otherwise: parent
        parent_or_size: Vec<i32>,
    }

    impl Dsu {
        pub fn new(size: usize) -> Self {
            Self {
                n: size,
                parent_or_size: vec![-1; size],
            }
        }

        pub fn merge(&mut self, a: usize, b: usize) -> usize {
            assert!(a < self.n);
            assert!(b < self.n);
            let (mut x, mut y) = (self.leader(a), self.leader(b));
            if x == y {
                return x;
            }
            if -self.parent_or_size[x] < -self.parent_or_size[y] {
                std::mem::swap(&mut x, &mut y);
            }
            self.parent_or_size[x] += self.parent_or_size[y];
            self.parent_or_size[y] = x as i32;
            x
        }

        pub fn same(&mut self, a: usize, b: usize) -> bool {
            assert!(a < self.n);
            assert!(b < self.n);
            self.leader(a) == self.leader(b)
        }

        pub fn leader(&mut self, a: usize) -> usize {
            assert!(a < self.n);
            if self.parent_or_size[a] < 0 {
                return a;
            }
            self.parent_or_size[a] = self.leader(self.parent_or_size[a] as usize) as i32;
            self.parent_or_size[a] as usize
        }

        pub fn size(&mut self, a: usize) -> usize {
            assert!(a < self.n);
            let x = self.leader(a);
            -self.parent_or_size[x] as usize
        }

        pub fn groups(&mut self) -> Vec<Vec<usize>> {
            let mut leader_buf = vec![0; self.n];
            let mut group_size = vec![0; self.n];
            for i in 0..self.n {
                leader_buf[i] = self.leader(i);
                group_size[leader_buf[i]] += 1;
            }
            let mut result = vec![Vec::new(); self.n];
            for i in 0..self.n {
                result[i].reserve(group_size[i]);
            }
            for i in 0..self.n {
                result[leader_buf[i]].push(i);
            }
            result
                .into_iter()
                .filter(|x| !x.is_empty())
                .collect::<Vec<Vec<usize>>>()
        }
    }
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
