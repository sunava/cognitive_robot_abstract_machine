"""Move an mp4's moov atom in front of mdat so a browser can read metadata and seek
without downloading the whole file. Chunk offsets in stco/co64 are shifted accordingly."""
import struct, sys, os

CONTAINERS={b'moov',b'trak',b'mdia',b'minf',b'stbl',b'edts',b'mvex',b'udta'}

def atoms(buf, start=0, end=None):
    end=len(buf) if end is None else end
    off=start
    while off+8<=end:
        size=struct.unpack('>I',buf[off:off+4])[0]
        typ=bytes(buf[off+4:off+8])
        hdr=8
        if size==1:
            size=struct.unpack('>Q',buf[off+8:off+16])[0]; hdr=16
        elif size==0:
            size=end-off
        yield off,size,typ,hdr
        off+=size

def shift_offsets(moov, delta):
    m=bytearray(moov)
    def walk(start,end):
        for off,size,typ,hdr in atoms(m,start,end):
            if typ in CONTAINERS:
                walk(off+hdr, off+size)
            elif typ==b'stco':
                n=struct.unpack('>I',m[off+hdr+4:off+hdr+8])[0]
                base=off+hdr+8
                for k in range(n):
                    p=base+4*k
                    struct.pack_into('>I',m,p,struct.unpack('>I',m[p:p+4])[0]+delta)
            elif typ==b'co64':
                n=struct.unpack('>I',m[off+hdr+4:off+hdr+8])[0]
                base=off+hdr+8
                for k in range(n):
                    p=base+8*k
                    struct.pack_into('>Q',m,p,struct.unpack('>Q',m[p:p+8])[0]+delta)
    walk(0,len(m))
    return bytes(m)

def convert(src,dst):
    buf=open(src,'rb').read()
    top=[(o,s,t,h) for o,s,t,h in atoms(buf)]
    names=[t.decode() for _,_,t,_ in top]
    if names.index(b'moov'.decode())<names.index(b'mdat'.decode()):
        print('already faststart'); return False
    moov=next((o,s) for o,s,t,_ in top if t==b'moov')
    keep=[(o,s,t) for o,s,t,_ in top if t not in (b'moov',b'free')]
    head=[(o,s,t) for o,s,t in keep if t==b'ftyp']
    rest=[(o,s,t) for o,s,t in keep if t!=b'ftyp']
    new_moov_at=sum(s for _,s,_ in head)
    old_mdat=next(o for o,_,t in rest if t==b'mdat')
    new_mdat=new_moov_at+moov[1]+sum(s for o,s,t in rest if o<old_mdat and t!=b'mdat')
    delta=new_mdat-old_mdat
    out=bytearray()
    for o,s,_ in head: out+=buf[o:o+s]
    out+=shift_offsets(buf[moov[0]:moov[0]+moov[1]], delta)
    for o,s,_ in rest: out+=buf[o:o+s]
    open(dst,'wb').write(out)
    print(f'{os.path.basename(src)} -> {os.path.basename(dst)}  moov moved to {new_moov_at}, offsets +{delta}, {len(out)/1e6:.1f} MB')
    return True

if __name__=='__main__':
    convert(sys.argv[1],sys.argv[2])
