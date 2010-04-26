# Back-Propagation Neural Networks
# Written in Ruby with NArray. See http://narray.rubyforge.org/
#
# This program is a translation of bpnn.py written by Neil Schemenauer <nas@arctrix.com>
# The original source is available at http://python.ca/nas/

require "narray"

# calculate a random number where:  a <= rand < b
class NArray
  def random_range!(a, b)
    random!(b-a) + a
  end
end

class NN
  def initialize(ni, nh, no)
    # number of input, hidden, and output nodes
    ni_ = ni + 1 # +1 for bias node
    nh_ = nh + 1 # +1 for bias node

    # activations for nodes
    @ai = NArray.float(ni_).fill!(1.0)
    @ah = NArray.float(nh_).fill!(1.0)
    @ao = NArray.float(no).fill!(1.0)

    # create weights and set them to random vaules
    @wi = NArray.float(nh,ni_).random_range!(-0.2, 0.2)
    @wo = NArray.float(no,nh_).random_range!(-2.0, 2.0)

    # last change in weights for momentum
    @ci = NArray.float(nh,ni_)
    @co = NArray.float(no,nh_)
  end

  def update(inputs)
    abort 'wrong number of inputs' if inputs.size != @ai.size-1

    # input activations
    #@ai[0] = NMath::tanh(inputs) #tanh
    @ai[0] = inputs #linear

    # hidden activations
    @ah[0] = 1.0/(1.0+NMath::exp(-(@wi.transpose(1,0).mul_add(@ai,0)))) #sigmoid
    #@ah[0] = NMath::tanh(@wi.transpose(1,0).mul_add(@ai,0)) #tanh

    # output activations
    @ao = @wo.transpose(1,0).mul_add(@ah,0) #linear
    #@ao = 1.0/(1.0+NMath::exp(-(@wo.transpose(1,0).mul_add(@ah,0)))) #sigmoid
    #@ao = NMath::tanh(@wo.transpose(1,0).mul_add(@ah,0)) #tanh

    return @ao
  end

  def backPropagate(targets, n, m)
    abort 'wrong number of target values' if targets.size != @ao.size

    # calculate error terms for output
    output_deltas = targets-@ao #linear
    #output_deltas = @ao*(1.0-@ao)*(targets-@ao) #sigmoid
    #output_deltas = (1.0-@ao**2)*(targets-@ao) #tanh

    # calculate error terms for hidden
    hidden_deltas = @ah*(1.0-@ah)*(output_deltas.mul_add(@wo,0)) #sigmoid
    #hidden_deltas = (1.0-@ah**2)*(output_deltas.mul_add(@wo,0)) #tanh

    # update output weights
    changes = @ah.newdim(0)*output_deltas
    @wo = @wo + n*changes + m*@co
    @co = changes

    # update input weights
    changes = @ai.newdim(0)*(hidden_deltas[0...-1])
    @wi = @wi + n*changes + m*@ci
    @ci = changes

    # calculate error
    return ((targets-@ao)**2).mul_add(0.5,0)
  end

  def test(patterns)
    # patterns.each{|p| puts "#{p[0]}\t#{update(p[0])}"}
    patterns.each{|p| puts "#{p[0].join("\t")}\t#{o=update(p[0]).to_a}\terror:#{p[1].zip(o).inject(0.0){|error, x| error + 0.5*(x[0]-x[1])**2}}"}
  end

  def weights
    puts 'Input weights:'
    p @wi
    puts
    puts 'Output weights:'
    p @wo
  end

  def train(patterns, iterations=1000, n=0.5, m=0.1)
    # n: learning rate
    # m: momentum factor
    patterns_nas = patterns.map{|pattern| pattern.map{|pa| NArray.to_na(pa)} }
    iterations.times do |i|
      error = 0.0
      patterns_nas.each do |pattern|
        inputs, targets = pattern
        update(inputs)
        error = error + backPropagate(targets, n, m)
      end
      puts "error %-14f" % error if i % 100 == 0
    end
  end
end

### main
if __FILE__ == $0
  pat = []
  f=open(ARGV[0])
  while line = f.gets
    pat << line.strip.split("\t").map{|d| d.split(",").map{|s|s.to_f} }
  end
  pat2 = []
  f=open(ARGV[1])
  while line = f.gets
    pat2 << line.strip.split("\t").map{|d| d.split(",").map{|s|s.to_f} }
  end

  10.times do |i|
    p i
    n = NN.new(pat[0][0].size, pat[0][0].size+i, pat[0][1].size)
    n.train(pat)
    n.test(pat2)
  end
end
