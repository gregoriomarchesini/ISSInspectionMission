clear
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% USER DEFINED PARAMETERS 
% PRO orbit parameters
rho_r   =  [50,64,78]; % m
rho_s   =  [0,0,0];   % m
rho_w   =  [0,60,140];  % m
alpha_r =  [deg2rad(90),deg2rad(90),deg2rad(90)];% rad
alpha_w =  [deg2rad(0),deg2rad(0),deg2rad(0)]; % rad 
k1      = 1.4;
k2      = 1.4;

DeltaT  = 0.1; % [s] sampling time of the ZOH MPC scheme
epsilon_u = 0.02; % [m/s2] maximum control acceleration

% barrier functions coefficients 
epsilon_r = 7;
epsilon_v = 0.133;
pdr0      = 0.02;
pdr1      = 0.05;
pdv0      = 0.05;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Standard parameters 
mu_earth = 3.98e14; % m^3/s^2
R_earth  = 6378e3;  % m
J2       = 0.0010826266835531513;   % m5/s2
kj2      = 3*R_earth^2*mu_earth*J2/2;
surface_density = 1.225;
    
%% ISS orbit Parameters 
ecc = 4.58e-4; %[-] eccentricity
p   = 6.80e6;  %[m] semimajor axis
angular_momentum = sqrt(p*mu_earth*(1-ecc^2)); %[m2/s] specific angular momenum
inc = deg2rad(55.99) ;% [rad] orbit inclination
mean_motion = angular_momentum/p^2;
period = 2*pi*sqrt(p^3/mu_earth);

% ISS aerodynamic parameters 
A_ch_drag = 1.3e2;   % m^2
mass_ch   = 4.196e5 ;% kg
Cd_ch     = 2.2;
CD_ch     = 0.5*A_ch_drag*Cd_ch/mass_ch;

% inspector aerodynamic  parameters 
A_dep_drag = 6e-3; % m^2
mass_dep   = 10 ;  % kg
Cd_dep     = 2.2;
CD_dep     = 0.5*A_dep_drag*Cd_dep/mass_dep;

% chief circular orbit altitude
h_ch      =  417000; % m
r_ch      =  R_earth + h_ch;
V_chief   = sqrt(mu_earth/r_ch);
density   = 3*10^(-12);          %kg/m^3 3 grams per km^3

%% Computing Maximum accelerations computation due to J2 and Drag
syms r [1,3]
U_pm = -mu_earth/norm(r);
U_j2 = -kj2/norm(r)^3*(1/3 - (r(3)/norm(r))^2);

nabla2U_pm = hessian(U_pm,r);
nabla2U_j2 = hessian(U_j2,r);

% transform to matlab function to a single vectorial variable anonymus
% function
nabla2U_pm = matlabFunction(nabla2U_pm); nabla2U_pm = @(r)nabla2U_pm(r(1),r(2),r(3));
nabla2U_j2 = matlabFunction(nabla2U_j2); nabla2U_j2 = @(r)nabla2U_j2(r(1),r(2),r(3));


C3 = @(x) [cos(x),-sin(x),0;
    sin(x),cos(x),0;
    0,    0,      1];
C1 = @(x) [1,     0,     0;
    0,cos(x),-sin(x);
    0,sin(x),cos(x)];

C313      = @(i,nu_bar)eye(3)*C1(i)*C3(nu_bar); % it is a rotation matrix
r_tilde   = @(nu_bar) [p*(1-ecc^2)/(1+ecc*cos(nu_bar)),0,0]';
r         = @(nu_bar) C313(inc,nu_bar)*r_tilde(nu_bar);

nabla2U_pm = @(nu_bar) norm(nabla2U_pm(r(nu_bar)),2);
nabla2U_j2 = @(nu_bar) norm(nabla2U_j2(r(nu_bar)),2);

nu_range = linspace(0,2*pi,300);

max_nabla2U_pm = max(arrayfun(nabla2U_pm,nu_range));
max_nabla2U_j2 = max(arrayfun(nabla2U_j2,nu_range));

%%
figure
sgtitle("maximum norm of the Hessian for J2 and point mass gravity potentila")
subplot(211);
grid on
hold on
plot(nu_range,arrayfun(nabla2U_pm,nu_range))
xlabel("true longitude [rad]")
ylabel("\nabla^2 U_{pm}")

subplot(212);
grid on
hold on
plot(nu_range,arrayfun(nabla2U_j2,nu_range))
xlabel("true longitude [rad]")
ylabel("\nabla^2 U_{J2}")


for jj = length(rho_r):-1:1
    
    
    %% PRO orbit computation 
    % PRO trajectory norm
    PRO_position     = @(t)                 vecnorm([rho_r(jj)*sin(mean_motion*t + alpha_r(jj)) ; rho_s(jj) + 2*rho_r(jj)*cos(mean_motion*t + alpha_r(jj));rho_w(jj)*sin(mean_motion*t + alpha_w(jj))]);
    PRO_velocity     = @(t) mean_motion   * vecnorm([rho_r(jj)*cos(mean_motion*t + alpha_r(jj)) ; -2*rho_r(jj)*sin(mean_motion*t + alpha_r(jj))           ;rho_w(jj)*cos(mean_motion*t + alpha_w(jj))]);
    PRO_acceleration = @(t) mean_motion^2 * vecnorm([-rho_r(jj)*sin(mean_motion*t + alpha_r(jj)); -2*rho_r(jj)*cos(mean_motion*t + alpha_r(jj))          ;-rho_w(jj)*sin(mean_motion*t + alpha_w(jj))]);
    sampling_interval = 5; %[s]
    sampling_points   = floor(period/sampling_interval);
    time = linspace(0,period,sampling_points);
    % compute max pos/vel/acc (found by simple search)
    [max_acc_ref,id_acc] = max(PRO_acceleration(time)); 
    [max_vel_ref,id_vel] = max(PRO_velocity(time));
    [max_pos_ref,id_pos] = max(PRO_position(time));
    % compute time of maximum pos/vel/acc 
    t_max_acc = time(id_acc);
    t_max_pos = time(id_pos);
    t_max_vel = time(id_vel);
    
       
    % aerodynamic drag maximum acceleration
    max_differential_drag = abs(density*(CD_dep - CD_ch)*V_chief^2);
    % J2 maximum acceleration 
    max_differential_j2 = k1*max_pos_ref * max_nabla2U_j2;
    % maximumm perturbing acceleration 
    epsilon_d = max_differential_j2 + max_differential_drag;
    
    % maximum relative dynamic acceleration
    omega_bar = angular_momentum/((1-ecc)*p)^2;
    epsilon_f = 2*omega_bar*k2*max_vel_ref + omega_bar^2*k1*max_pos_ref + k1*max_pos_ref * max_nabla2U_pm;
    
    %% Computation of Lipschitz constants
    
    a_bar     = epsilon_f + epsilon_u + epsilon_d;
    a_barZero = epsilon_f + epsilon_d; % when no control is active
    
    epsilon_vbar = epsilon_v + (a_bar+max_acc_ref)*DeltaT;
    epsilon_rbar = epsilon_r + (a_bar+max_acc_ref)*DeltaT^2/2 + (epsilon_v + max_vel_ref)*DeltaT;
    
    epsilon_vbarZero = epsilon_v + (a_barZero+max_acc_ref)*DeltaT;
    epsilon_rbarZero = epsilon_r + (a_barZero+max_acc_ref)*DeltaT^2/2 + (epsilon_v + max_vel_ref)*DeltaT;
    
    % constants under active actutation 
    L_dv_l = 2*a_bar^2 + 2*pdv0*epsilon_vbar*a_bar;
    c_dv_l = 2*epsilon_vbar;
    L_dr_l = 6*a_bar*epsilon_vbar + 2*(pdr1+pdr0)*(epsilon_vbar^2 + epsilon_rbar*a_bar) + 2*pdr1*pdr0*(epsilon_rbar*epsilon_vbar);
    c_dr_l = 2*epsilon_rbar;
    % constants under zero actutation 
    L_dv_lZero = 2*a_barZero + 2*pdv0 *epsilon_vbarZero*a_barZero;
    c_dv_lZero = 2*epsilon_vbarZero;
    L_dr_lZero = 6*a_barZero + 2*(pdr1+pdr0)*(epsilon_vbarZero^2 + epsilon_rbarZero*a_barZero) + 2*pdr1*pdr0*(epsilon_rbarZero*epsilon_vbarZero);
    c_dr_lZero = 2*epsilon_rbarZero;
    
    
    % save seetings to file
    settings(jj).L_dv_l = L_dv_l;
    settings(jj).L_dr_l = L_dr_l;
    settings(jj).c_dv_l = c_dv_l;
    settings(jj).c_dr_l = c_dr_l;
    settings(jj).L_dv_lZero = L_dv_lZero;
    settings(jj).L_dr_lZero = L_dr_lZero;
    settings(jj).c_dv_lZero = c_dv_lZero;
    settings(jj).c_dr_lZero = c_dr_lZero;
    settings(jj).DeltaT = DeltaT;
    settings(jj).epsilon_f = epsilon_f;
    settings(jj).epsilon_r = epsilon_r;
    settings(jj).epsilon_v = epsilon_v;
    settings(jj).epsilon_rbar = epsilon_rbar;
    settings(jj).epsilon_vbar = epsilon_vbar;
    settings(jj).epsilon_d = epsilon_d;
    settings(jj).pdr0      = pdr0;
    settings(jj).pdr1      = pdr1;
    settings(jj).pdv0      = pdv0;
    settings(jj).a_bar = a_bar;
    settings(jj).max_acc_ref = max_acc_ref;
    settings(jj).max_vel_ref = max_vel_ref;
    settings(jj).u_max = epsilon_u;

    directory = sprintf("../simulationsResult/agent%i",jj-1);
    mkdir(directory)
    writetable(struct2table(settings(jj)), fullfile(directory,'settings.csv'))
end
%%

fileID = fopen('multiagentParameters.txt','w');
fieldsName = fields(settings);

for jj= 1:length(fieldsName)
      row = fprintf(fileID,'%s & %.3e & %.3e & %.3e \\\\ \n',string(fieldsName{jj}),settings(1).(fieldsName{jj}),settings(2).(fieldsName{jj}),settings(3).(fieldsName{jj}));
end
fclose(fileID);





