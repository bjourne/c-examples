// Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
//
// smallpt, a Path Tracer originally written by Kevin Beason, 2008. I
// have modified it to fit my coding style and also to make it a bit
// faster.
//
// Compile with:
//
//     g++ -o smallpt smallpt.cpp -O3 -march=native -mtune=native
//
// Add -fopenmp for multi-threading.
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

struct Vec {
    double x, y, z;
    Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; }
    Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); }
    Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); }
    Vec operator*(double b) const { return Vec(x*b,y*b,z*b); }
    Vec mult(const Vec &b) const {
        return Vec(x*b.x, y*b.y, z*b.z);
    }
    Vec& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
    double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // cross:
    Vec operator%(Vec&b) {
        return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
    }
};

static inline Vec
v_add(Vec a, Vec b) {
    return Vec(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline Vec
v_sub(Vec a, Vec b) {
    return Vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline double
v_dot(Vec a, Vec b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct Ray {
    Vec o, d;
    Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

enum Refl_t {
    DIFF, SPEC, REFR
};

struct Sphere {
    double rad_sq;
    Vec p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)

    Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_):
        p(p_), e(e_), c(c_), refl(refl_) {
        rad_sq = rad_ * rad_;
    }
};

Sphere spheres[] = {
    Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left
    Sphere(1e5, Vec(-1e5+99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght
    Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back
    Sphere(1e5, Vec(50,40.8,-1e5+170), Vec(),Vec(),           DIFF),//Frnt
    Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm
    Sphere(1e5, Vec(50,-1e5+81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top
    Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, SPEC),//Mirr
    Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas
    Sphere(600, Vec(50,681.6-.27,81.6),Vec(12,12,12),  Vec(), DIFF) //Lite
};
#define N_SPHERES 9

static inline double
sph_intersect(Vec pos, double rad_sq, Vec ro, Vec rd) {
    Vec op = v_sub(pos, ro);
    double eps = 1e-4;
    double b = v_dot(op, rd);

    double det = b * b - v_dot(op, op) + rad_sq;
    if (det < 0)
        return 0.0;
    det = sqrt(det);

    double t = b - det;
    if (t > eps)
        return t;
    t = b + det;
    if (t > eps)
        return t;
    return 0.0;
}

inline double
clamp(double x){
    return x < 0 ? 0 : x > 1 ? 1 : x;
}
inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); }

static inline bool
intersect(Vec ro, Vec rd, double &t, int &id) {
    double inf = t = 1e20;
    for(int i = N_SPHERES; i--;) {
        Vec p = spheres[i].p;
        double rad_sq = spheres[i].rad_sq;
        double d = sph_intersect(p, rad_sq, ro, rd);
        if (d && d < t) {
            t = d;
            id = i;
        }
    }
    return t < inf;
}
Vec radiance(const Ray &r,
             int depth,
             unsigned short *Xi) {
    double t;
    int id = 0;
    if (!intersect(r.o, r.d, t, id))
        return Vec();
    const Sphere &obj = spheres[id];        // the hit object
    Vec x = r.o + r.d *t;
    Vec n = (x - obj.p).norm();
    Vec nl = n.dot(r.d) < 0 ? n : n * -1;
    Vec f = obj.c;
    double p = f.x>f.y && f.x>f.z ? f.x : f.y > f.z ? f.y : f.z;
    if (++depth > 5) {
        if (erand48(Xi) < p)
            f = f  * (1.0 / p);
        else
            return obj.e; //R.R.
    }
    if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
        double r1 = 2 * M_PI * erand48(Xi);
        double r2 = erand48(Xi);
        double r2s = sqrt(r2);
        Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u;
        Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
        return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));
    } else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
        return obj.e + f.mult(radiance(Ray(x, r.d-n*2*n.dot(r.d)), depth,Xi));
    Ray reflRay(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION
    bool into = n.dot(nl)>0;                // Ray from outside going in?
    double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.d.dot(nl), cos2t;
    if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection
        return obj.e + f.mult(radiance(reflRay,depth,Xi));
    Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
    double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n));
    double Re = R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);

    Vec mulVec;
    if (depth > 2) {
        if (erand48(Xi) <P) {
            mulVec = radiance(reflRay, depth, Xi)*RP;
        } else {
            mulVec = radiance(Ray(x, tdir), depth, Xi) * TP;
        }
    } else {
        mulVec = radiance(reflRay, depth, Xi) * Re + radiance(Ray(x, tdir), depth, Xi) * Tr;
    }
    return obj.e + f.mult(mulVec);
}

int main(int argc, char *argv[]) {
    int w = 1024;
    int h = 768;
    int samps = 4;
    Ray cam(Vec(50, 52, 295.6),
            Vec(0, -0.042612, -1).norm());
    Vec cx = Vec(w * .5135 / h);
    Vec cy = (cx % cam.d).norm()*.5135, r;
    Vec *c = new Vec[w*h];
    //#pragma omp parallel for schedule(dynamic, 1) private(r)
    for (int y = 0; y < h; y++) {
        for (unsigned short x = 0, Xi[3] = {0, 0, (unsigned short)(y*y*y)}; x<w; x++) {
            for (int sy=0, i=(h-y-1)*w+x; sy<2; sy++) {
                for (int sx = 0; sx < 2; sx++, r = Vec()) {
                    for (int s = 0; s < samps; s++) {
                        double r1 = 2 * erand48(Xi);
                        double dx = r1 < 1 ? sqrt(r1) - 1
                                         : 1 - sqrt(2 - r1);
                        double r2 = 2 * erand48(Xi);
                        double dy = r2 < 1 ? sqrt(r2) - 1
                                         : 1 - sqrt(2 - r2);
                        Vec d = cx * (((sx + .5 + dx)/2 + x)/w - .5) +
                                         cy*( ( (sy+.5 + dy)/2 + y)/h - .5) + cam.d;
                        r = r + radiance(Ray(cam.o+d*140, d.norm()), 0, Xi) * (1./ samps);
                    } // Camera rays are pushed ^^^^^ forward to start in interior
                    c[i].x += clamp(r.x) * 0.25;
                    c[i].y += clamp(r.y) * 0.25;
                    c[i].z += clamp(r.z) * 0.25;
                }
            }
        }
    }
    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i=0; i<w*h; i++) {
        fprintf(f,"%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
    }
}
