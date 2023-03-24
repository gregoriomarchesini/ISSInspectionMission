clear all
close all


path_list = ["./agent0","./agent1","./agent2"];
ax_list = cell(1,5);

cbf_fig    = figure();
cbf_fig_ax = gca();
ax_list2   = cell(1,5);

for jj = 1:5
    ax_list{jj} = subplot(5,1,jj);
    grid on
    hold(ax_list{jj})
    ax_list{jj}.XLim = [0,3];
end
lwidth      = 1; % linewidth for all the plots
for kk = 1:length(path_list) 

    CBF         = readtable(fullfile(path_list(kk),'CBF_data_frame.csv'));
    control     = readtable(fullfile(path_list(kk),'control_data_frame.csv'));
    state       = readtable(fullfile(path_list(kk),'state_data_frame.csv'));
    state_ref   = readtable(fullfile(path_list(kk),'reference_state_data_frame.csv'));
    CBF.time    = CBF.time/60 ;
    
    % plot the CBF performace
    
    % velocity CBF
   
    plot(ax_list{1},CBF.time,CBF.CBF_velocity,LineWidth=lwidth,DisplayName=['inspector ',num2str(kk)]);
    ax_list{1}.XLabel.String = "time [min]";
    ax_list{1}.YLabel.String = "h_{\delta v}";
    ax_list{1}.YLim = [-0.03,max(CBF.CBF_velocity)*1.2];
    % position CBF
    
    plot(ax_list{2},CBF.time,CBF.CBF_position,LineWidth=lwidth);
    ax_list{2}.XLabel.String = "time [min]";
    ax_list{2}.YLabel.String = "h_{\delta r}";
    ax_list{2}.YLim = [-0.03,max(CBF.CBF_position)*1.2];
    % position HOBF
    
    plot(ax_list{3},CBF.time,CBF.velocity_constraint,LineWidth=lwidth);
    ax_list{3}.XLabel.String = "time [min]";
    ax_list{3}.YLabel.String = "\zeta_{\delta v}";
    ax_list{3}.YLim = [-0.003,max(CBF.velocity_constraint)*1.2];
    
    % state performace
    state_array           = table2array(state);
    disp(state_array(1,:))
    reference_state_array = table2array(state_ref);
    state_error = state_array-reference_state_array;
    
    pos_error = vecnorm(state_error(:,1:3)');
    vel_error = vecnorm(state_error(:,4:end)');

    plot(ax_list{4},CBF.time,CBF.position_constraint,LineWidth=lwidth);
    ax_list{4}.XLabel.String = "time [min]";
    ax_list{4}.YLabel.String = "\zeta_{\delta r}";
    ax_list{4}.YLim = [-0.03,max(CBF.position_constraint)*1.2];

    control.u_norm = sqrt(control.u_y.^2 + control.u_z.^2 + control.u_x.^2);
    stairs(ax_list{5},CBF.time,control.u_norm,LineWidth=lwidth);
    ax_list{5}.XLabel.String = "time [min]";
    ax_list{5}.YLabel.String = "||u||_2";
    ax_list{5}.YLim = [0,0.03];
    yline(0.02,"--")

end
legend(ax_list{1})


