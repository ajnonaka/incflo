#include <AMReX_Vector.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_ParmParse.H>

#include <incflo.H>

using namespace amrex;

void incflo::init_bcs ()
{
    has_inout_bndry = false;

    m_bcrec_velocity.resize(AMREX_SPACEDIM);
    m_bcrec_density.resize(1);
    if (m_ntrac > 0) { m_bcrec_tracer.resize(m_ntrac); }

    auto f = [this] (std::string const& bcid, Orientation ori)
    {
        m_bc_density[ori] = 1.0;
        AMREX_D_TERM(m_bc_velocity[ori][0] = 0.0;, // default
                     m_bc_velocity[ori][1] = 0.0;,
                     m_bc_velocity[ori][2] = 0.0;);
        m_bc_tracer[ori].resize(m_ntrac,0.0);

        ParmParse pp(bcid);
        std::string bc_type_in = "null";
        pp.query("type", bc_type_in);
        std::string bc_type = amrex::toLower(bc_type_in);

        if (bc_type == "pressure_inflow" || bc_type == "pi")
        {
            amrex::Print() << bcid << " set to pressure inflow.\n";

            m_bc_type[ori] = BC::pressure_inflow;

            pp.get("pressure", m_bc_pressure[ori]);
            pp.queryarr("tracer", m_bc_tracer[ori], 0, m_ntrac);
            // Set mathematical BCs here also
            AMREX_D_TERM(m_bcrec_velocity[0].set(ori, BCType::foextrap);,
                         m_bcrec_velocity[1].set(ori, BCType::foextrap);,
                         m_bcrec_velocity[2].set(ori, BCType::foextrap););
            m_bcrec_density[0].set(ori, BCType::foextrap);
            //when the VOF method is used, the default BC for tracer (i.e., the keyword 'tracer'
            //is not explicitly included in the bcid) is symmetrical.
            if ( pp.contains("tracer") ) {
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::ext_dir); }
            }else if(m_vof_advect_tracer){
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::reflect_even); }
            }
            else {
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::foextrap); }
            }
        }
        else if (bc_type == "pressure_outflow" || bc_type == "po")
        {
            amrex::Print() << bcid << " set to pressure outflow.\n";

            m_bc_type[ori] = BC::pressure_outflow;

            pp.get("pressure", m_bc_pressure[ori]);
            pp.queryarr("tracer", m_bc_tracer[ori], 0, m_ntrac);
            // Set mathematical BCs here also
            AMREX_D_TERM(m_bcrec_velocity[0].set(ori, BCType::foextrap);,
                         m_bcrec_velocity[1].set(ori, BCType::foextrap);,
                         m_bcrec_velocity[2].set(ori, BCType::foextrap););
            // Only normal oriection has reflect_even
            //for (int dim = 0; dim < AMREX_SPACEDIM; dim++){
                //if (dim !=ori.coordDir())
                 // m_bcrec_velocity[ori.coordDir()].set(ori, BCType::reflect_even);
                //else
                // m_bcrec_velocity[dim].set(ori, BCType::ext_dir);
            //}
            m_bcrec_density[0].set(ori, BCType::foextrap);
            //when the VOF method is used, the default BC for tracer (i.e., the keyword 'tracer'
            //is not explicitly included in the bcid) is symmetrical.
            if ( pp.contains("tracer") ) {
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::ext_dir); }
            }else if(m_vof_advect_tracer){
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::reflect_even); }
            }
            else {
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::foextrap); }
            }


        }
        else if (bc_type == "mass_inflow" || bc_type == "mi")
        {
            amrex::Print() << bcid << " set to mass inflow.\n";

            m_bc_type[ori] = BC::mass_inflow;

            std::vector<Real> v;
            if (pp.queryarr("velocity", v, 0, AMREX_SPACEDIM)) {
               for (int i=0; i<AMREX_SPACEDIM; i++){
                   m_bc_velocity[ori][i] = v[i];
               }
            }

            pp.query("density", m_bc_density[ori]);
            pp.queryarr("tracer", m_bc_tracer[ori], 0, m_ntrac);

            // Set mathematical BCs
            AMREX_D_TERM(m_bcrec_velocity[0].set(ori, BCType::ext_dir);,
                         m_bcrec_velocity[1].set(ori, BCType::ext_dir);,
                         m_bcrec_velocity[2].set(ori, BCType::ext_dir););
            m_bcrec_density[0].set(ori, BCType::ext_dir);
            for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::ext_dir); }
        }
        else if (bc_type == "direction_dependent" || bc_type == "dd" )
        {
            amrex::Print() << bcid << " set to direction-dependent.\n";

            has_inout_bndry = true;

            m_bc_type[ori] = BC::direction_dependent;

            std::vector<Real> v;
            if (pp.queryarr("velocity", v, 0, AMREX_SPACEDIM)) {
               for (int i=0; i<AMREX_SPACEDIM; i++){
                   m_bc_velocity[ori][i] = v[i];
               }
            }

            pp.query("density", m_bc_density[ori]);
            pp.queryarr("tracer", m_bc_tracer[ori], 0, m_ntrac);

            AMREX_D_TERM(m_bcrec_velocity[0].set(ori, BCType::direction_dependent);,
                         m_bcrec_velocity[1].set(ori, BCType::direction_dependent);,
                         m_bcrec_velocity[2].set(ori, BCType::direction_dependent););
            m_bcrec_density[0].set(ori, BCType::direction_dependent);
            for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::direction_dependent); }
        }
        else if (bc_type == "no_slip_wall" || bc_type == "nsw")
        {
            amrex::Print() << bcid <<" set to no-slip wall.\n";

            m_bc_type[ori] = BC::no_slip_wall;

            // Note that m_bc_velocity defaults to 0 above so we are ok if
            //      queryarr finds nothing
            // Here we make sure that we only use the tangential components
            //      of a specified velocity field -- the wall is not allowed
            //      to move in the normal direction
            std::vector<Real> v;
            if (pp.queryarr("velocity", v, 0, AMREX_SPACEDIM)) {
                v[ori.coordDir()] = 0.0;
                for (int i=0; i<AMREX_SPACEDIM; i++){
                    m_bc_velocity[ori][i] = v[i];
                }
            }

            // We potentially read in values at no-slip walls in the event that the
            // tracer has Dirichlet bcs
            pp.queryarr("tracer", m_bc_tracer[ori], 0, m_ntrac);

            // Set mathematical BCs
            AMREX_D_TERM(m_bcrec_velocity[0].set(ori, BCType::ext_dir);,
                         m_bcrec_velocity[1].set(ori, BCType::ext_dir);,
                         m_bcrec_velocity[2].set(ori, BCType::ext_dir););
            m_bcrec_density[0].set(ori, BCType::foextrap);
            //when the VOF method is used, the default BC for tracer (i.e., the keyword 'tracer'
            //is not explicitly included in the bcid) is symmetrical.
            if ( pp.contains("tracer") ) {
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::ext_dir); }
            }else if(m_vof_advect_tracer){
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::reflect_even); }
            } else {
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::foextrap); }
            }
        }
        else if (bc_type == "slip_wall" || bc_type == "sw")
        {
            amrex::Print() << bcid <<" set to slip wall.\n";

            m_bc_type[ori] = BC::slip_wall;

            // These values are set by default above -
            //      note that we only actually use the zero value for the normal direction.
            // m_bc_velocity[ori] = {0.0, 0.0, 0.0};

            // We potentially read in values at slip walls in the event that the
            // tracer has Dirichlet bcs
            pp.queryarr("tracer", m_bc_tracer[ori], 0, m_ntrac);

            // Tangential directions have hoextrap
            AMREX_D_TERM(m_bcrec_velocity[0].set(ori, BCType::hoextrap);,
                         m_bcrec_velocity[1].set(ori, BCType::hoextrap);,
                         m_bcrec_velocity[2].set(ori, BCType::hoextrap););
            // Only normal oriection has ext_dir
            m_bcrec_velocity[ori.coordDir()].set(ori, BCType::ext_dir);
            if (m_advection_type == "BDS") {
                // BDS requires foextrap to avoid introduction of local max/min
                m_bcrec_density[0].set(ori, BCType::foextrap);
            } else{
                m_bcrec_density[0].set(ori, BCType::hoextrap);
            }
            if ( pp.contains("tracer") ) {
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::ext_dir); }
            } else {
                for (auto& b : m_bcrec_tracer) {
                    if (m_advection_type == "BDS") {
                        b.set(ori, BCType::foextrap);
                    } else {
                        b.set(ori, BCType::hoextrap);
                    }
                }
            }
        }
        else if (bc_type == "mixed" )
        {
            amrex::Print() << bcid << " set to mixed inflow outflow.\n";
            m_has_mixedBC = true;
#ifdef AMREX_USE_EB
            // ReadParameters() already called
            if (m_advection_type != "Godunov") { amrex::Abort("mixed BCs require Godunov"); }

            ParmParse ipp("incflo");
            std::string eb_geom = "null";
            ipp.query("geometry", eb_geom);
            eb_geom = amrex::toLower(eb_geom);
            if (eb_geom == "null" || eb_geom == "all_regular")
#endif
            {
                Abort("For now, mixed BCs must be separated by an EB");
            }
            Warning("Using BC type mixed requires that the Dirichlet and Neumann regions are separated by EB.");

            m_bc_type[ori] = BC::mixed;

            pp.get("pressure", m_bc_pressure[ori]);

            std::vector<Real> v;
            if (pp.queryarr("velocity", v, 0, AMREX_SPACEDIM)) {
               for (int i=0; i<AMREX_SPACEDIM; i++){
                   m_bc_velocity[ori][i] = v[i];
               }
            }

            pp.query("density", m_bc_density[ori]);
            pp.queryarr("tracer", m_bc_tracer[ori], 0, m_ntrac);

            // Set mathematical BCs. BC_mask will handle Dirichlet part.
            AMREX_D_TERM(m_bcrec_velocity[0].set(ori, BCType::foextrap);,
                         m_bcrec_velocity[1].set(ori, BCType::foextrap);,
                         m_bcrec_velocity[2].set(ori, BCType::foextrap););
            m_bcrec_density[0].set(ori, BCType::foextrap);
            for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::foextrap); }
        }
        else
        {
            m_bc_type[ori] = BC::undefined;
        }

        if (geom[0].isPeriodic(ori.coordDir())) {
            if (m_bc_type[ori] == BC::undefined) {
                m_bc_type[ori] = BC::periodic;

                // Set mathematical BCs
                AMREX_D_TERM(m_bcrec_velocity[0].set(ori, BCType::int_dir);,
                             m_bcrec_velocity[1].set(ori, BCType::int_dir);,
                             m_bcrec_velocity[2].set(ori, BCType::int_dir););
                m_bcrec_density[0].set(ori, BCType::int_dir);
                for (auto& b : m_bcrec_tracer) { b.set(ori, BCType::int_dir); }
            } else {
                amrex::Abort("Wrong BC type for periodic boundary");
            }
        }
    };

    f("xlo", Orientation(Direction::x,Orientation::low));
    f("xhi", Orientation(Direction::x,Orientation::high));
    f("ylo", Orientation(Direction::y,Orientation::low));
    f("yhi", Orientation(Direction::y,Orientation::high));
#if (AMREX_SPACEDIM == 3)
    f("zlo", Orientation(Direction::z,Orientation::low));
    f("zhi", Orientation(Direction::z,Orientation::high));
#endif

    if (m_ntrac > 0) {
        Vector<Real> h_data(m_ntrac*AMREX_SPACEDIM*2);
        Real* hp = h_data.data();
        for (auto const& v : m_bc_tracer) {
            for (auto x : v) {
                *(hp++) = x;
            }
        }

        m_bc_tracer_raii.resize(m_ntrac*AMREX_SPACEDIM*2);
        Real* p = m_bc_tracer_raii.data();
#ifdef AMREX_USE_GPU
        Gpu::htod_memcpy
#else
        std::memcpy
#endif
            (p, h_data.data(), sizeof(Real)*h_data.size());

        for (int i = 0; i < AMREX_SPACEDIM*2; ++i) {
            m_bc_tracer_d[i] = p;
            p += m_ntrac;
        }
    }

    // Copy BCRecs to device container
    m_bcrec_velocity_d.resize(AMREX_SPACEDIM);
#ifdef AMREX_USE_GPU
    Gpu::htod_memcpy
#else
    std::memcpy
#endif
        (m_bcrec_velocity_d.data(), m_bcrec_velocity.data(), sizeof(BCRec)*AMREX_SPACEDIM);

    m_bcrec_density_d.resize(1);
#ifdef AMREX_USE_GPU
    Gpu::htod_memcpy
#else
    std::memcpy
#endif
        (m_bcrec_density_d.data(), m_bcrec_density.data(), sizeof(BCRec));

    if (m_ntrac > 0)
    {
        m_bcrec_tracer_d.resize(m_ntrac);
#ifdef AMREX_USE_GPU
        Gpu::htod_memcpy
#else
        std::memcpy
#endif
            (m_bcrec_tracer_d.data(), m_bcrec_tracer.data(), sizeof(BCRec)*m_ntrac);
    }

    // force
    {
        const int ncomp = std::max(m_ntrac, AMREX_SPACEDIM);
        m_bcrec_force.resize(ncomp);
        for (OrientationIter oit; oit; ++oit) {
            Orientation ori = oit();
            int dir = ori.coordDir();
            Orientation::Side side = ori.faceDir();
            auto const bct = m_bc_type[ori];
            if (bct == BC::periodic)
            {
                if (side == Orientation::low) {
                    for (auto& b : m_bcrec_force) b.setLo(dir, BCType::int_dir);
                } else {
                    for (auto& b : m_bcrec_force) b.setHi(dir, BCType::int_dir);
                }
            }
            else
            {
                if (side == Orientation::low) {
                    for (auto& b : m_bcrec_force) b.setLo(dir, BCType::foextrap);
                } else {
                    for (auto& b : m_bcrec_force) b.setHi(dir, BCType::foextrap);
                }
            }
        }
        m_bcrec_force_d.resize(ncomp);
#ifdef AMREX_USE_GPU
        Gpu::htod_memcpy
#else
        std::memcpy
#endif
            (m_bcrec_force_d.data(), m_bcrec_force.data(), sizeof(BCRec)*ncomp);

    }
}

