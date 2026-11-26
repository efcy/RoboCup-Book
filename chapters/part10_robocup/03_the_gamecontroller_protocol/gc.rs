use std::convert::TryInto;

const HEADER: &[u8; 4] = b"RGme";
const VERSION: u8 = 20;
pub const CONTROL_LEN: usize = 158;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum State { Initial, Ready, Set, Playing, Finished }

#[derive(Debug, Clone, Copy, Default)]
pub struct Player { pub penalty: u8, pub secs_left: u8, pub cautions: u8 }

#[derive(Debug, Clone)]
pub struct Team {
    pub number: u8,
    pub score: u8,
    pub goalkeeper: u8,
    pub message_budget: u16,
    pub players: [Player; 20],
}

#[derive(Debug, Clone)]
pub struct Control {
    pub packet_number: u8,
    pub stopped: bool,
    pub state: State,
    pub set_play: u8,
    pub first_half: bool,
    pub kicking_team: u8,
    pub secs_remaining: i16,
    pub secondary_time: i16,
    pub teams: [Team; 2],
}

fn parse_team(b: &[u8]) -> Team {
    // TeamInfo is 70 bytes: 6 x u8, u16 singleShots, u16 budget, 20 x 3 bytes
    let mut players = [Player::default(); 20];
    for (i, p) in players.iter_mut().enumerate() {
        let o = 10 + 3 * i;
        *p = Player { penalty: b[o], secs_left: b[o + 1], cautions: b[o + 2] };
    }
    Team {
        number: b[0],
        score: b[4],
        goalkeeper: b[3],
        message_budget: u16::from_le_bytes([b[8], b[9]]),
        players,
    }
}

pub fn parse_control(buf: &[u8]) -> Result<Control, String> {
    if buf.len() != CONTROL_LEN { return Err(format!("length {}", buf.len())); }
    if &buf[0..4] != HEADER { return Err("bad header".into()); }
    if buf[4] != VERSION { return Err(format!("version {}", buf[4])); }
    let state = match buf[10] {
        0 => State::Initial, 1 => State::Ready, 2 => State::Set,
        3 => State::Playing, 4 => State::Finished,
        s => return Err(format!("unknown state {s}")),
    };
    Ok(Control {
        packet_number: buf[5],
        stopped: buf[8] == 1,
        state,
        set_play: buf[11],
        first_half: buf[12] == 1,
        kicking_team: buf[13],
        secs_remaining: i16::from_le_bytes(buf[14..16].try_into().unwrap()),
        secondary_time: i16::from_le_bytes(buf[16..18].try_into().unwrap()),
        teams: [parse_team(&buf[18..88]), parse_team(&buf[88..158])],
    })
}

pub fn encode_return(player: u8, team: u8, fallen: bool, pose: [f32; 3],
                     ball_age: f32, ball: [f32; 2]) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[0..4].copy_from_slice(b"RGrt");
    b[4] = 4;
    b[5] = player;
    b[6] = team;
    b[7] = fallen as u8;
    let floats = [pose[0], pose[1], pose[2], ball_age, ball[0], ball[1]];
    for (i, f) in floats.iter().enumerate() {
        b[8 + 4 * i..12 + 4 * i].copy_from_slice(&f.to_le_bytes());
    }
    b
}

fn main() {
    let ctl = std::fs::read("control.bin").unwrap();
    let c = parse_control(&ctl).unwrap();
    println!("{:?} stopped={} set_play={} first_half={} kicking={} secs={} sec2={}",
        c.state, c.stopped, c.set_play, c.first_half, c.kicking_team,
        c.secs_remaining, c.secondary_time);
    for t in &c.teams {
        println!("team {} score {} gk {} budget {} p3={:?}", t.number, t.score,
            t.goalkeeper, t.message_budget, t.players[2]);
    }
    let want = std::fs::read("return.bin").unwrap();
    let got = encode_return(3, 12, false, [-2500.0, 800.0, 0.5], 1.5, [900.0, -120.0]);
    println!("return packet identical to the C struct: {}", want == got);
    println!("short packet: {:?}", parse_control(&ctl[..100]).err());
    let mut bad = ctl.clone(); bad[4] = 19;
    println!("old version: {:?}", parse_control(&bad).err());
}
