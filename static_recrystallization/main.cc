
#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  prismspf::MPIInitFinalize mpi_init(argc, argv);
  
  constexpr unsigned int dim    = 2;
  constexpr unsigned int degree = 1;

  // Parse the command line options (if there are any) to get the name of the input
  // file
  ParseCMDOptions cli_options(argc, argv);

  // Read the user input parameters
  UserInputParameters<dim> user_inputs(cli_options.get_parameters_filename());

  unsigned int num_parent_grains = user_inputs.user_constants.get_int("num_parent_grains");
  unsigned int num_nucleating_grains = user_inputs.user_constants.get_int("num_nucleating_grains");
  unsigned int num_op_total = num_parent_grains + num_nucleating_grains;
  unsigned int first_postprocess_index = num_op_total + num_parent_grains;

  // Declare field attributes
  std::vector<FieldAttributes> field_attributes = {};
  std::vector<unsigned int> nucleating_field_indices = {};
  unsigned int field_count = 0;

  // Add order parameters for parent grains
  for (unsigned int i = 0; i < num_parent_grains; ++i)
    {
      field_count++;
      std::string field_name = "n" + std::to_string(field_count);
      field_attributes.push_back(FieldAttributes(field_name, Scalar));
    }

  // Add order parameters for recrystallized nuclei
  for (unsigned int i = 0; i < num_nucleating_grains; ++i)
    {
      nucleating_field_indices.push_back(field_count);
      field_count++;
      std::string field_name = "n" + std::to_string(field_count);
      field_attributes.push_back(FieldAttributes(field_name, Scalar));
    }

  // Stored energy fields for the parent grains
  unsigned int rho_count = 0;
  for (unsigned int i = 0; i < num_parent_grains; ++i)
    {
      field_count++;
      rho_count++;
      std::string field_name = "rho" + std::to_string(rho_count);
      field_attributes.push_back(FieldAttributes(field_name, Scalar));
    }

  // Declare postprocessing fields
  field_attributes.push_back(FieldAttributes("sum_op_squared", Scalar));
  field_attributes.push_back(FieldAttributes("sum2op", Scalar));
  field_attributes.push_back(FieldAttributes("rho", Scalar));
  
  // Nucleation rate
  FieldAttributes nucl_rate = FieldAttributes("nucleation_rate", Scalar);
  nucl_rate.is_nucleation_rate_variable = true;
  nucl_rate.nucleating_field_indices = nucleating_field_indices;
  field_attributes.push_back(nucl_rate);

  // Now that the field attributes are declared,
  // create solve blocks: one for grains, then one for postprocessing
  std::set<Types::Index> exp_indices;
  std::set<Types::Index> rho_indices;
  std::set<std::string> exp_dependencies;
  std::set<std::string> rho_dependencies;
  std::set<std::string> pp_dependencies;
  std::set<std::string> nucl_dependencies;
  for (unsigned int i = 0; i < num_op_total; ++i)
    {
      exp_indices.insert(i);
      exp_dependencies.insert("old_1(n" + std::to_string(i+1) + ")");
      exp_dependencies.insert("grad(old_1(n" + std::to_string(i+1) + "))");
      pp_dependencies.insert("n" + std::to_string(i+1));
    }
  
  for (unsigned int i = 0; i < num_parent_grains; ++i)
    {
      rho_indices.insert(i + num_op_total);
      exp_dependencies.insert("old_1(rho" + std::to_string(i+1) + ")");
      rho_dependencies.insert("old_1(rho" + std::to_string(i+1) + ")");
      pp_dependencies.insert("rho" + std::to_string(i+1));
    }
  
  for (unsigned int i = 0; i < num_parent_grains; ++i)
    {
      nucl_dependencies.insert("n" + std::to_string(i+1));
    }
  nucl_dependencies.insert("rho");
  
  // Solve blocks: order parameters, postprocssing fields, and nucleation rate
  SolveBlock exp_block;
  exp_block.id            = 1;
  exp_block.solve_type    = Explicit;
  exp_block.solve_timing  = Primary;
  exp_block.field_indices = exp_indices;
  exp_block.dependencies_rhs =
    make_dependency_set(field_attributes, exp_dependencies);

  SolveBlock rho_block;
  rho_block.id              = 2;
  rho_block.solve_type      = Explicit;
  rho_block.solve_timing    = Primary;
  rho_block.field_indices   = rho_indices;
  rho_block.dependencies_rhs =
    make_dependency_set(field_attributes, rho_dependencies);

  SolveBlock pp_block;
  pp_block.id               = 3;
  pp_block.solve_type       = Explicit;
  pp_block.solve_timing     = PostProcess;
  pp_block.field_indices    = {first_postprocess_index, first_postprocess_index + 1, first_postprocess_index + 2};
  pp_block.dependencies_rhs = make_dependency_set(field_attributes, pp_dependencies);

  SolveBlock nucl_block;
  nucl_block.id             = 4;
  nucl_block.solve_type     = Explicit;
  nucl_block.solve_timing   = NucleationRate;
  nucl_block.field_indices  = {first_postprocess_index + 3};
  nucl_block.dependencies_rhs = make_dependency_set(field_attributes, nucl_dependencies);

  std::vector<SolveBlock> solve_blocks({exp_block, rho_block, pp_block, nucl_block});

  // Initialize, then solve, the phase-field PDEs (runs the simulation)
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(field_attributes,
                                       solve_blocks,
                                       user_inputs,
                                       pf_tools,
                                       pde_operator);
  problem.solve();

  return 0;
}
