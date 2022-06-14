clear;  clc; close all;

x = -3:0.01:3;

ySigm = 1./(1+exp(-x));
yTanh = tanh(x);
yReLU = max(0,x);

alpha = .1;
yLeakyReLU=min(alpha*x,0)+max(0,x);


ySoftPlus =log(1+exp(x));
ySoftmax = exp(x)/(sum(exp(x)));  % a = softmax(n) = exp(n)/sum(exp(n))




 figure(1)

   hold on; grid off; box on;
   set(figure(1), 'Position', [500   100   500   500])

    ..................................................................................................
     plot(x, ySigm,'b','LineWidth',2);
     plot(x, ySoftPlus,'g','LineWidth',2);
     
     
     plot(x, yTanh,'r','LineWidth',2);
     plot(x, yReLU,':r','LineWidth',2);
     plot(x, yLeakyReLU,'--k','LineWidth',2);
     plot(0, 0,'w');

     axis([-3 3 -3 3])
     
%           fill([x flip(x)],[ones(size(x)) zeros(size(x))],'b','LineStyle','none','FaceColor','b','FaceAlpha',.3)
%           fill([x flip(x)],[ones(size(x)) -ones(size(x))],'r','LineStyle','none','FaceColor','r','FaceAlpha',.1)

     plot([0 0],[-3 3],'-k')
     plot([-3 3],[-0 0],'-k')
 
     set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10)                
     

     
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';

..................................................................................................
  Legend = {'Sigmoid:    $f(x)=1/({1+e^{-x}})$';
            'Softplus:   $f(x)=log_e({1+e^{x}})$';
            'Tanh:       $f(x)=2/(1+e^{-2x})-1$';
            'ReLU:       $f(x)=max(0,x)$';
            'Leaky ReLU: $f(x)$=min$(\alpha x,0)$+max$(0,x)$'};
%   Legend = {'Leaky ReLU: $f(x)$=min$(\alpha x,0)$+max$(0,x)$'};

% plot([0 0],[-3 3],'w')
% plot([-3 3],[-0 0],'w')
 axis([-3 3 -3 3])

    text(-2.85,-2.03,Legend,...
        'HorizontalAlignment','left','fontsize',13,...
        'fontname','Times New Roman','FontWeight','Bold',...
        'EdgeColor','w','BackgroundColor','w','LineWidth',1,'interpreter','latex')

      legend('Sigmoid','Softplus','Tanh','ReLU','Leaky ReLU','    (\alpha=0.1)','fontsize',13','location','northwest')
      
      
      
      
      
      



 
