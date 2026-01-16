-- schema.sql
PRAGMA foreign_keys = ON;

-- table elements
CREATE TABLE elements (
  elem TEXT PRIMARY KEY,
  ref_state TEXT,
  atomic_mass REAL,
  h298_h0 REAL,
  s298 REAL
);

-- table functions
CREATE TABLE functions (
  func TEXT PRIMARY KEY,
  elem TEXT NOT NULL,
  temp_start REAL,
  temp_end REAL,
  expression TEXT,
  is_continued TEXT CHECK (is_continued IN ('Y', 'N')),
  FOREIGN KEY (elem) REFERENCES elements(elem) -- func is for simple element
);

-- table tdbs
CREATE TABLE tdbs (
  tdb TEXT PRIMARY KEY,
  description TEXT,  
  version TEXT NOT NULL DEFAULT '1.0',
  update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- table phases
CREATE TABLE phases (
  phase TEXT,
  sub_lattices INTEGER,
  stoichiometry TEXT,
  components TEXT,
  tdb TEXT,
  PRIMARY KEY (phase, tdb),
  FOREIGN KEY (tdb) REFERENCES tdbs(tdb)
);

-- table parameters
CREATE TABLE parameters (
  param TEXT,
  ptype TEXT CHECK (ptype IN ('G', 'L')),
  phase TEXT,
  components TEXT,
  order_num INTEGER CHECK (order_num >= 0),
  temp_start REAL,
  temp_end REAL,
  expression TEXT,
  is_continued TEXT CHECK (is_continued IN ('Y', 'N')),
  tdb TEXT,
  PRIMARY KEY ("param", tdb),
  FOREIGN KEY (phase, tdb) REFERENCES phases(phase, tdb)
);
