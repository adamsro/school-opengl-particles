typedef float4 point;typedef float4 vector;typedef float4 color;typedef float4 sphere;constant float4 G       = (float4) ( 0., -9.8, 0., 0. );constant float  DT      = 0.1; // time stepconstant sphere Sphere1 = (sphere)( -100., -800., 0.,  600. ); // x, y, z, rvector Bounce( vector in, vector n ) {    n.w = 0.;    n = normalize( n );    vector out = in - 2.*n*dot(in.xyz, n.xyz);    out.w = 0.;    return out;}vector BounceSphere( point p, vector v, sphere s ){	// calculate normal of sphere at point	vector n = fast_normalize(p-s);	return Bounce(v, n);}bool IsInsideSphere( point p, sphere s ){	// if the distance between the point and the sphere's center is less than the radius than true.	if(fast_distance(p.xyz, s.xyz) <= s.w) {		return true;	}	return false;}kernel void Particle( global point *dPobj, global vector *dVel, global color *dCobj ){    int gid = get_global_id( 0 );    point  p = dPobj[gid];    vector v = dVel[gid];    point  pp = p + v*DT + .5*DT*DT*G;    vector vp = v + G*DT;    pp.w = 1.;    vp.w = 0.;    if( IsInsideSphere( pp, Sphere1 ) )    {        vp = BounceSphere( p, v, Sphere1 );        pp = p + vp*DT + .5*DT*DT*G;    }    dPobj[gid] = pp;    dVel[gid]  = vp;}