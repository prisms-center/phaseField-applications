// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue = dealii::VectorizedArray<number>;
  using ScalarGrad  = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorValue = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad  = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess  = dealii::Tensor<3, dim, ScalarValue>;
  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  /**
   * @brief Constructor.
   */
  CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
    : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools)
    , m_well(get_user_inputs().user_constants.get_double("m_well"))
    , kappa(get_user_inputs().user_constants.get_double("kappa"))
    , alpha(get_user_inputs().user_constants.get_double("alpha"))
    , kinetic_coef(get_user_inputs().user_constants.get_double("kinetic_coef"))
    , fsbs(get_user_inputs().user_constants.get_double("fsbs"))
    , r_nuc(get_user_inputs().user_constants.get_double("r_nuc"))
    , r_freeze(get_user_inputs().user_constants.get_double("r_freeze"))
    , seeding_duration(get_user_inputs().user_constants.get_double("seeding_duration"))
    , interface_coef(std::sqrt(2.0 * kappa / m_well))
    , A(get_user_inputs().user_constants.get_double("seeding_coef"))
    , num_parent_grains(get_user_inputs().user_constants.get_int("num_parent_grains"))
    , num_nucleating_grains(get_user_inputs().user_constants.get_int("num_nucleating_grains"))
    , seeding_increment(get_user_inputs().user_constants.get_int("seeding_increment"))
  {
    num_op_total = num_parent_grains + num_nucleating_grains;
    first_postprocess_index = num_op_total + num_parent_grains;
  }

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    // Initialize all variables to zero that are not loaded from a file
    scalar_value = 0.0;
  }

  void
  compute_rhs(FieldContainer<dim, degree, number> &variable_list,
              const SimulationTimer               &sim_timer,
              unsigned int                         solve_block_id) const override
  {
    if (solve_block_id == 1) // explicit n
      {
        // Calculate sum of squares of the order parameters
        ScalarValue sum_op_sq = 0.0;
        ScalarValue rho_calc = 0.0;
        ScalarValue eq_n;
        ScalarGrad eqx_n;

        // Calculating the terms required for the governing equations
        for (unsigned int i = 0; i < num_op_total; i++)
          {
            ScalarValue n_val  = variable_list.template get_value<Scalar, OldOne>(i);
            sum_op_sq += n_val * n_val;
          }

        for (unsigned int i = 0; i < num_parent_grains; i++) // Only the parent grains have nonzero rho
          {
            ScalarValue n_val = variable_list.template get_value<Scalar, OldOne>(i);
            ScalarValue rhoi  = variable_list.template get_value<Scalar, OldOne>(num_op_total + i);
            rho_calc += n_val * n_val * rhoi / (sum_op_sq + 1e-12);
          }

        for (unsigned int i = 0; i < num_op_total; i++)
          {
            ScalarValue n_val  = variable_list.template get_value<Scalar, OldOne>(i);
            ScalarGrad  n_grad = variable_list.template get_gradient<Scalar, OldOne>(i);
            ScalarValue rhoi   = 0.0;
            if (i < num_parent_grains)
              {
                rhoi = variable_list.template get_value<Scalar, OldOne>(num_op_total + i);
              }

            // Nucleation expressions - source_term and gamma are modified by seed_nucleus()
            // Default behavior with no nuclei: no source term and gamma=1.0 multipies the RHS
            // With nuclei: the source term is finite, and gamma is reduced to slow/freeze
            // the area around the newly-placed nucleus
            ScalarValue source_term(0.0);
            ScalarValue gamma(1.0);
            seed_nucleus(variable_list.get_q_point_location(), source_term, gamma, sim_timer, i);

            ScalarValue fnV = -n_val + (n_val * n_val * n_val);
            for (unsigned int j = 0; j < num_op_total; j++)
              {
                if (i != j)
                  {
                    ScalarValue nj_val  = variable_list.template get_value<Scalar, OldOne>(j);
                    fnV += 2.0 * alpha * n_val * nj_val * nj_val;
                  }
              }
            fnV *= m_well;
            ScalarValue fsV = fsbs * 2.0 * n_val * (rhoi - rho_calc) / (sum_op_sq + 1e-12);

            eq_n = n_val - sim_timer.get_timestep() * gamma * kinetic_coef * (fnV + fsV);
            eqx_n = -sim_timer.get_timestep() * gamma * kinetic_coef * kappa * n_grad;
            
            variable_list.set_value_term(i, eq_n + source_term);
            variable_list.set_gradient_term(i, eqx_n);
          }
      }
    else if (solve_block_id == 2) // rho_i
      {
        for (unsigned int i = num_op_total; i < num_parent_grains + num_op_total; i++)
          {
            ScalarValue rho_val = variable_list.template get_value<Scalar, OldOne>(i);
            
            // Static recovery would go here.
            // Define an expression for the rate of change of rho:
            ScalarValue drho_dt = 0.0;

            variable_list.set_value_term(i, rho_val + sim_timer.get_timestep() * drho_dt);
          }
      }
    else if (solve_block_id == 3) // postprocess
      {
        // Loop over all OPs
        ScalarValue sum_op_squared = 0.0;
        ScalarValue sum2op = 0.0;
        ScalarValue rho_calc = 0.0;
        for (unsigned int i = 0; i < num_op_total; i++)
          {
            ScalarValue n_val  = variable_list.template get_value<Scalar, Current>(i);
            sum_op_squared += n_val * n_val;
            for (unsigned int j = i + 1; j < num_op_total; j++)
              {
                ScalarValue n2_val  = variable_list.template get_value<Scalar, Current>(j);
                sum2op += n_val * n2_val;
              }
          }
        for (unsigned int i = 0; i < num_parent_grains; i++) // Only the parent grains have nonzero rho
          {
            ScalarValue n_val = variable_list.template get_value<Scalar, Current>(i);
            ScalarValue rhoi  = variable_list.template get_value<Scalar, Current>(num_op_total + i);
            rho_calc += n_val * n_val * rhoi / (sum_op_squared + 1e-12);
          }
        
        variable_list.set_value_term(first_postprocess_index, sum_op_squared);
        variable_list.set_value_term(first_postprocess_index + 1, sum2op);
        variable_list.set_value_term(first_postprocess_index + 2, rho_calc);
      }
    else if (solve_block_id == 4) // Nucleation rate
      {
        ScalarValue nucProb = 1e-30;
        if (sim_timer.get_increment() == seeding_increment)
          {
            ScalarValue rho = variable_list.template get_value<Scalar, Current>(first_postprocess_index + 2);
            ScalarValue sum2op = 0.0;

            for (unsigned int i = 0; i < num_parent_grains; i++)
              {
                ScalarValue n = variable_list.template get_value<Scalar, Current>(i);
                for (unsigned int j = i + 1; j < num_parent_grains; j++)
                  {
                    ScalarValue n2 = variable_list.template get_value<Scalar, Current>(j);
                    sum2op += n * n2;
                  }
              }
            
            for (unsigned int v = 0; v < ScalarValue::size(); v++)
              {
                if (sum2op[v] > 0.15)
                  {
                    nucProb[v] = std::exp(-(M_PI / 4) * std::pow(33790 / (A * std::sqrt(rho[v] * 1.0e6)), 2));
                  }
              }
          }
        
        variable_list.set_value_term(first_postprocess_index + 3, nucProb);
      }
  }

  void seed_nucleus(const dealii::Point<dim, ScalarValue> &q_point_loc,
                    ScalarValue                           &source_term,
                    ScalarValue                           &gamma,
                    const SimulationTimer                 &sim_timer,
                    unsigned int                          field_index) const
  {
    unsigned int current_increment = sim_timer.get_increment();
    double current_time = sim_timer.get_time();
    // Iterate through nuclei list
    for (const Nucleus<dim> &nucleus : get_pf_tools().nuclei_list)
      {
        if (nucleus.field_index == field_index)
          {
            // Calculate the distance function to the nucleus center
            const dealii::Point<dim, ScalarValue> loc_as_arr = [&]()
            {
              dealii::Point<dim, ScalarValue> result;
              const dealii::Point<dim>       &point = nucleus.location;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  result[d] = ScalarValue(point[d]);
                }
              return result;
            }();

            ScalarValue dist =
              get_user_inputs().spatial_discretization.distance(q_point_loc, loc_as_arr);
            
            // TODO: re-write this part
            // Seed a nucleus if it was added to the list of nuclei recently
            if (current_time < nucleus.seed_time + seeding_duration)
              {
                gamma *= 0.5 * (1.0 + std::tanh((dist - r_freeze) / interface_coef));
              }
            if (nucleus.seed_increment == current_increment - 1)
              {
                source_term += 0.5 * (1.0 - std::tanh((dist - r_nuc) / interface_coef));
              }
          }
      }
  }

  number m_well;
  number kappa;
  number alpha;
  number kinetic_coef;
  number fsbs;
  number r_nuc;
  number r_freeze;
  number interface_coef;
  number A;
  unsigned int num_parent_grains;
  unsigned int num_nucleating_grains;
  unsigned int num_op_total;
  unsigned int first_postprocess_index;
  unsigned int seeding_increment;
  double seeding_duration;
};

PRISMS_PF_END_NAMESPACE
